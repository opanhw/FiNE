from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook
from ...util.generate import generate_fast

from .fine_hparams import FINEHyperParams
from .utils import get_top_neurons
import torch.nn.functional as F
from .repr_tools import get_words_idxs_in_templates


CONTEXT_TEMPLATES_CACHE = None


def apply_fine_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: FINEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs,
):
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
    torch.manual_seed(42)

    if copy:
        model = deepcopy(model)

    weights_copy = {}

    deltas, model_top_neurons = [], []

    for idx, single_request in enumerate(request):

        single_deltas, single_model_top_neurons, _weights_copy = execute_fine(model, tok, single_request, hparams, idx, **kwargs)

        deltas.append(single_deltas)
        model_top_neurons.append(single_model_top_neurons)

        if return_orig_weights:
            weights_copy.update(_weights_copy)

    deltas = torch.cat(deltas, dim=0)
    model_top_neurons = torch.cat(model_top_neurons, dim=0)

    if kwargs.get('find_only') is not None and kwargs['find_only']:
        return model, weights_copy

    # modify model state dict
    state_dict = model.state_dict()

    for j, neuron in enumerate(model_top_neurons):
        delta = deltas[j]
        w = state_dict[f'{hparams.rewrite_module_tmp.format(neuron[0])}.weight'].T
        w[neuron[1]] += delta.to(w.device)
        state_dict[f'{hparams.rewrite_module_tmp.format(neuron[0])}.weight'] = w.T

    # load new state dict
    model.load_state_dict(state_dict)

    if return_orig_weights:
        weights_copy = _weights_copy

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def execute_fine(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: FINEHyperParams,
    request_idx: int,
    **kwargs,
):
    """
    Executes the FiNE update algorithm
    """

    # Update target and print info
    request = deepcopy(request)
    if request["target_new"] != " ":
        # Space required for correct tokenization
        request["target_new"] = " " + request["target_new"]

    if '{}' not in request['prompt']:
        assert request['subject'] in request['prompt'] or \
               print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

        request['prompt'] = request['prompt'].replace(request['subject'], '{}')

    print(
        f"Executing FINE algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
    )

    print(f"Finding neurons...")

    if "llama-2" in hparams.model_name.lower():
        target_ids = tok.encode(request['target_new'].strip(), add_special_tokens=False, return_tensors="pt")
    else:
        target_ids = tok.encode(request['target_new'], add_special_tokens=False, return_tensors="pt")
    if hparams.add_eos:
        target_ids = torch.cat([target_ids, torch.tensor([[tok.eos_token_id]])], dim=-1)
    

    target_len = target_ids.shape[-1]
    print(target_ids[0].tolist())

    prompt_ids = tok.encode(request['prompt'].format(request['subject']), add_special_tokens=True, return_tensors="pt")
    if model.config.model_type == "gptj":
        prompt_ids = torch.cat([torch.tensor([[tok.eos_token_id]], dtype=torch.long), prompt_ids], dim=-1)


    input_ids = torch.cat([prompt_ids, target_ids], dim=-1).cuda()

    print(tok.batch_decode(input_ids, skip_special_tokens=False)[0])

    model.eval()
    for param in model.named_parameters():
        param[1].requires_grad = False

    model.set_get_activations()

    with torch.no_grad():
        model_outputs = model.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids).cuda(),
        )

    model.cal_activations = False

    original_distribution = torch.cat([model_outputs.logits[0, :-target_len - 1]], dim=0)
    original_distribution_last = model_outputs.logits[0, -1:]

    model_outputs_ids = model_outputs.logits.argmax(-1).cpu()
    original_output_ids = model_outputs_ids[0, -target_len - 1:-1]
    activations = torch.stack(model.activations).permute(1, 2, 0, 3)[0, -target_len - 1:-1]
    print(activations.shape)

    if not hparams.random:
        model_top_neurons = get_top_neurons(hparams, model, activations, original_output_ids, hparams.neuron_num, specific_layer=hparams.layer)

    else:
        print("Randomly selected top_neurons")
        model_top_neurons = torch.zeros(target_len, hparams.neuron_num, 2, dtype=torch.int16)
        for i, x in enumerate(model_top_neurons):
            for j, y in enumerate(x):
                if hparams.layer is None:
                    model_top_neurons[i, j, 0] = torch.randint(0, activations.shape[-2] - hparams.last_layer, (1, ))[0]
                else:
                    model_top_neurons[i, j, 0] = hparams.layer
                model_top_neurons[i, j, 1] = torch.randint(0, activations.shape[-1], (1, ))[0]

    
    print(model_top_neurons)

    model_top_neurons = model_top_neurons.reshape(-1, 2)

    neurons_dict = {}

    for i, neuron in enumerate(model_top_neurons):
        neuron = neuron.tolist()
        if neuron[0] not in neurons_dict:
            neurons_dict[neuron[0]] = {"loc": [neuron[1]], "idx": [i]}
        else:
            neurons_dict[neuron[0]]['loc'].append(neuron[1])
            neurons_dict[neuron[0]]['idx'].append(i)


    if kwargs.get('find_only') is not None and kwargs['find_only']:
        torch.save(model_top_neurons[:, :, :], kwargs['find_file_path'])
        return torch.tensor([]), torch.tensor([]), {}
    
    deltas = (torch.zeros(model_top_neurons.shape[0], model.config.hidden_size, dtype=torch.bfloat16)).cuda().requires_grad_(True)
    optimizer = torch.optim.Adam([deltas], lr=hparams.lr)
    
    def forward_hook(n):
        def fn(module, input, output):
            if n in neurons_dict:
                neuron_loc = neurons_dict[n]['loc']
                neuron_idx = neurons_dict[n]['idx']
                output_delta = input[0][:, :, neuron_loc].to(deltas.dtype) @ deltas[neuron_idx].requires_grad_(True)

                return output[0] + output_delta.to(input[0].dtype)

        return fn

    if model.config.model_type == "llama":
        handle = [model.model.layers[n].mlp.down_proj.register_forward_hook(forward_hook(n)) for n in
                  range(model.config.num_hidden_layers)]
    elif model.config.model_type == "gptj":
        handle = [model.transformer.h[n].mlp.fc_out.register_forward_hook(forward_hook(n)) for n in
                  range(model.config.num_hidden_layers)]


    print(sum([torch.cuda.max_memory_allocated(f"cuda:{i}") / 1024 ** 3 for i in range(torch.cuda.device_count())]))
    print("Start editing...")

    for epoch in range(hparams.epochs):
        model_outputs = model.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids).cuda(),
        )

        output = model_outputs.logits[0, -target_len-1:-1].float()

        nll_loss = F.nll_loss(F.log_softmax(output, dim=-1), target_ids[0, :].to(output.device), reduction="none").max()
        kl_loss = F.kl_div(F.log_softmax(model_outputs.logits[0, :-target_len - 1], dim=-1),
                           F.softmax(original_distribution, dim=-1), reduction='batchmean') * hparams.alpha
        penalty_loss = (-torch.log(1 - F.softmax(model_outputs.logits[0, -1].float(), dim=-1)[target_ids[0, :]])).max() * hparams.beta

        norm_loss = torch.norm(deltas, p=2) * hparams.gamma
        loss = nll_loss + norm_loss + kl_loss + penalty_loss

        if (epoch + 1) % 1 == 0:
            print(
                f"Request {request_idx} "
                f"Epoch {epoch + 1} Loss: {loss.item():.4f} NLL: {nll_loss.item():.4f} Norm: {norm_loss.item():.4f} KL: {kl_loss.item():.4f} Pen: {penalty_loss.item():.4f} "
                f"avg prob of [{request['target_new']}] "
                f"{torch.gather(F.softmax(output, dim=-1), -1, target_ids[0, :].unsqueeze(-1).to(output.device)).min().item():.8f} "
                f"output id: {model_outputs.logits[0, -target_len-1:-1].float().argmax(-1).tolist()} "
                f"{F.softmax(model_outputs.logits[0, -1].float(), dim=-1)[target_ids[0, :]].tolist()}"
            )
        
        if hparams.early_stop_prob is not None and nll_loss < -np.log(hparams.early_stop_prob) or np.isnan(nll_loss.detach().cpu()):
            break



        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    for h in handle:
        h.remove()

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in list(set(model_top_neurons[:, 0].flatten().tolist()))
    }

    print(weights.keys())

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    return deltas.detach(), model_top_neurons, weights_copy


def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x.replace("{", "").replace("}", "") + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["The", "Therefore", "Because", "I", "You"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
