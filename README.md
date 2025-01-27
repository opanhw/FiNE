# Precise Localization of Memories: A Fine-grained Neuron-level Knowledge Editing Technique for LLMs
This repository is the official implementation of the paper "Precise Localization of Memories: A Fine-grained Neuron-level Knowledge Editing Technique for LLMs" (ICLR 2025).

## Requirements

To get started, please clone this repository and install packages as:

```bash
git clone https://github.com/opanhw/FiNE.git
conda create -n FiNE python=3.11
...
pip install -r requirements.txt
```

## Benchmark

- KnowEdit Benchmark

See details of the benchmark in [this page](https://huggingface.co/datasets/zjunlp/KnowEdit).

## Editing

**Please setting parameters before editing**. Parameters of FiNE can be found in `./hparams/FINE/`, which contain:

- `model_name`: path to the model you choose
- `epochs`: maximum iterations
- `lr`: learning rate
- `alpha`: coefficient of KL divergence
- `beta`: coefficient of repetition penalty loss
- `gamma`: coefficient of norm loss
- `neuron_num`: number of edited neurons 
- `early_stop_prob`: minimum probability of early stopping
- `last_layer`: number of last frozen layers
- `fp16`: true for single precision, false for double precision

A sample editing command for gpt-j-6b is:

```bash
bash run_gptj_fine.sh
```

A sample editing command for llama-2-7b is:

```bash
bash run_llama2_fine.sh
```

A sample editing command for llama-3-8b is:

```bash
bash run_llama3_fine.sh
```

## Acknowledgements

Our code is based on [EasyEdit](https://github.com/zjunlp/EasyEdit) and [MM_Neurons](https://github.com/opanhw/MM_Neurons).

Thanks [@wonderful9462](https://github.com/wonderful9462) for the assistance in this work.

## Citation

If you find this code useful, please kindly cite our work as:

```bibtex

```

