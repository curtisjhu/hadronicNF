# Normalizing flow
Use NF to generate kinematic variables

# Installation
```bash
module load python
conda create --name <env_name> python=3.9 pip
# module load root
conda activate <env_name>
pip install -e .
```

# Instructions

### Hadronic NF
MC-Generation will output a .csv file, use csv_to_root to convert to a .root file.
Adjust parameters in config_nf_hadronic.yml
To run training and generation:
```bash
python train_cond_hadronic.py --config_file config_nf_hadronic.yml --log-dir <model_name> --data-dir data --epochs 500
```


# Config file
The configuration can be found in `config_nf.yml` where the following can be specified:
- file_name: list of paths to data files
- out_branch_names: branches to generate or used to calculate the quantity to generate
- truth_branch_names: branches with conditional inputs
- data_branch_names: branches with additional information

Hyperparameters:
- lr: 0.001
- batch_size: 512
- num_layers: 2
- latent_size: 128
- num_bijectors: 10
- hidden_activation

# Code
- preprocess.py: load and preprocess data
- made.py: model definition and construction
- utils.py: loss functions
- utils_plot.py: plotting functions
- generate.ipynb: stand-alone code for generation and plotting