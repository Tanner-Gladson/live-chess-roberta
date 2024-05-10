#!/bin/bash
#SBATCH --partition=shared
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:0
#SBATCH --mail-user=tannergladson@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name="chess-roberta-preprocess"
#SBATCH --output=slurm-%j.out
#SBATCH --mem=16G
#SBATCH -o ./slurm_output/preprocess-%A.out

# Environment configs
ENV_NAME="chess-roberta"
PYTHON_VERSION="3.9.18"

# run the actual script
echo "node list: "$SLURM_JOB_NODELIST
echo "master address: "$MASTER_ADDR

module load anaconda
if [[ $(conda env list | grep -w $ENV_NAME) ]]; then
    echo "Conda environment '$ENV_NAME' already exists, continueing"
else
    conda create --name $ENV_NAME python=$PYTHON_VERSION
fi
conda activate $ENV_NAME
pip install -r requirements.txt

python -u ./lichess_bot.py