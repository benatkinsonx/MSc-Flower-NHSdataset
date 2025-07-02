# Federated Learning and Federated Analytics with Flower

## Set Up

- clone the repository
- create the conda environment: `conda env create -f environment.yaml`

## Dataset

Dataset can be found on Kaggle at https://www.kaggle.com/datasets/utkarshx27/breast-cancer-dataset-used-royston-and-altman and is located in the repo at `data/gbsg.csv`

Features

- pid = patient ID
- age (years)
- meno = menopausal (0=pre-meno, 1=post-meno)
- size = tumour size in mm
- grade = tumour grade
- nodes = number of positive lymph nodes
- pgr = progesterone receptors (fmol/l)
- er = estrogen receptors (fmol/l)
- hormon = hormonal therapy (0=no, 1=yes)
- rfstime = recurrence free survival time (days)
- status = 0=alive without recurrence, 1=recurrence or death

## Federated Analytics (FA)
To run the FA simulation: `python src/FedAnalytics/simulation.py`. By default, this returns the mean age of the patients in the dataset.

Currently the only summary statistic that can be computed is the mean.

See `src/FedAnalytics/config.py` for the configuration parameters.

## Federated Learning (FL)
To run the FL simulation: `python src/FedLearning/simulation.py`. By default, this runs logistic regression binary classification task to predict the `status` feature.

Currently the only model is logistic regression for classifying the status.

See `src/FedLearning/config.py` for the configuration parameters.

### How to activate conda environment in GitBash terminal
- source path/to/miniconda3/etc/profile.d/conda.sh
- conda activate msc-project-env
