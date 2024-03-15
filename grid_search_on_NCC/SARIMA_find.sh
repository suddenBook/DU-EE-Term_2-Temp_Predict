#!/bin/bash

#SBATCH --job-name=SARIMA_params
#SBATCH --partition=cpu
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --qos=long-cpu
#SBATCH --mem=39G
#SBATCH --time=13-21:00:00

source /etc/profile

pip install --user --upgrade  pandas numpy matplotlib seaborn statsmodels scikit-learn tqdm

python3 find_param.py
