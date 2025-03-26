from dataclasses import dataclass
from typing import List
import yaml

from ...util.hparams import HyperParams


@dataclass
class FINEHyperParams(HyperParams):
    # Method
    epochs: int = None
    lr: float = None
    gamma: int = None
    beta: int = None
    alpha: int = None
    neuron_num: int = None
    layer: int = None
    add_eos: bool = False
    early_stop_prob: float = None
    random: bool = False
    add_random: bool = False
    layer_start: int = None
    layer_end: int = None
    last_layer: int = 0
    batch_size: int = 1

    stats_dir: str = None
    fact_token: str = None
    context_template_length_params: List = None
    mom2_adjustment: bool = False
    mom2_dataset: str = None
    mom2_n_samples: int = None
    mom2_dtype: str = None

    # Module templates
    layer_projection_tmp: str = None
    rewrite_module_tmp: str = None
    layer_module_tmp: str = None
    mlp_module_tmp: str = None
    attn_module_tmp: str = None
    ln_f_module: str = None
    lm_head_module: str = None

    # Statistics
    alg_name: str = None
    device: int = None
    model_name: str = None

    max_length: int = 40
    model_parallel: bool = False

    modified: bool = True
    fp16: bool = True
    padding: str = 'right'

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'FINE') or print(f'FINEHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
