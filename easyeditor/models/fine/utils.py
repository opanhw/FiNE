import torch
import copy
import random
import numpy as np
from tqdm import tqdm


def get_top_neurons(hparams, model, activations, original_output_ids, neuron_num, sample_neuron_num=None,
                     specific_layer=None, specific_layers=None):
    total_top_neurons = []
    for j, id in enumerate(original_output_ids):
        if specific_layers is not None:
            scores = torch.zeros((1, activations.shape[2])).half().cuda()
            act = activations[j, specific_layers[j]].cuda()
            if 'gpt-j' in hparams.model_name:
                down_proj = model.transformer.h[specific_layers[j]].mlp.fc_out.weight.data.T.half()
                unembedding = model.lm_head.weight.data.T.half()
            else:
                down_proj = model.model.layers[specific_layers[j]].mlp.down_proj.weight.data.T.half()
                unembedding = model.lm_head.weight.data.T.half()

            prob = act * ((down_proj.to(act.device) @ unembedding[:, id].to(act.device)).flatten())
            scores[0] += prob.flatten()

        elif specific_layer is not None:
            scores = torch.zeros((1, activations.shape[2])).half().cuda()
            act = activations[j, specific_layer].cuda()
            if 'gpt-j' in hparams.model_name:
                down_proj = model.transformer.h[specific_layer].mlp.fc_out.weight.data.T.half()
                unembedding = model.lm_head.weight.data.T.half()
            else:
                down_proj = model.model.layers[specific_layer].mlp.down_proj.weight.data.T.half()
                unembedding = model.lm_head.weight.data.T.half()

            prob = act * ((down_proj.to(act.device) @ unembedding[:, id].to(act.device)).flatten())
            scores[0] += prob.flatten()


        else:
            # calculate scores of neurons
            scores = torch.zeros((activations.shape[1] - hparams.last_layer, activations.shape[2])).half().cuda()
            unembedding = model.lm_head.weight.data.T[:, id].half()
            for layer in range(activations.shape[1] - hparams.last_layer):
                act = activations[j, layer].cuda()

                if 'gpt-j' in hparams.model_name:
                    down_proj = model.transformer.h[layer].mlp.fc_out.weight.data.T.half()
                else:
                    down_proj = model.model.layers[layer].mlp.down_proj.weight.data.T.half()
                

                prob = act * ((down_proj.to(act.device) @ unembedding.to(act.device)).flatten())
                scores[layer] += prob.flatten()

        _, top_index = torch.topk(scores.flatten(), neuron_num)

        if specific_layers is not None:
            top_pos = [(specific_layers[j], x.item()) for x in top_index]
        elif specific_layer is not None:
            top_pos = [(specific_layer, x.item()) for x in top_index]
        else:
            if sample_neuron_num:
                top_pos = random.sample([(x.item() // scores.shape[1], x.item() % scores.shape[1]) for x in top_index],
                                        sample_neuron_num)
            else:
                top_pos = [(x.item() // scores.shape[1], x.item() % scores.shape[1]) for x in top_index]

        top_neurons = []
        for i, pos in enumerate(top_pos):
            top_neurons.append([pos[0], pos[1]])

        total_top_neurons.append(top_neurons)

    top_neurons = torch.tensor(total_top_neurons)

    return top_neurons
