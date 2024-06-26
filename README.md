# MSDAFL
Molecular Substructure-based Dual Attention Feature Learning framework for Drug-Drug Interaction Prediction

## Requirements
* torch==2.1.0
* torch-geometric==2.4.0
* numpy==1.26.1
* rdkit==2023.9.1 
* scikit-learn==1.3.2
## File
### data
We use `ddi_get_mol_graphs.py` and `ddi_get_smiles.py` to process the raw data, dividing the drug pairs into training, validation, and testing sets in a ratio of 6:2:2. We run the experiments on three random folds.The statistics of the preprocessed datasets are listed as follows:
* The ZhangDDI dataset is of small-scale, consisting of 544 drugs and 45,720 pairwise DDIs.
* The ChCh-Miner dataset is of medium-scale, consisting of 997 drugs and 21,486 pairwise DDIs.
* The DeepDDI dataset is of large-scale, consisting of 1,704 drugs and 191,870 pairwise DDIs.
### core
* `ddi_layers.py` contains self-attention  and interactive attention encoder
* `ddi_predictor.py` contains GNN encoder and  nomorlization module
### run
Run by command line, e.g.:

`python ddi_main.py --device cuda --dataset ZhangDDI --hidden_dim 128 --num_patterns 60 --batch_size 512`

`python ddi_main.py --device cuda --dataset ChCh-Miner --hidden_dim 128 --num_patterns 60 --batch_size 512`

`python ddi_main.py --device cuda --dataset DeepDDI --hidden_dim 128 --num_patterns 60 --batch_size 512`

For detailed command line options, see `ddi_main.py`

## Contact
If you have any questions or comments, please feel free to email Cheng Yan(yancheng01@hnucm.edu.cn).