## RUN WITH source.sh


### CREATE CLAUDE ENVIRONMENT ###
# Create persistent conda environment (when working inside a pod)
# conda env create -f environment.yml --prefix .conda_env
# conda env update -f environment.yml --prefix .conda_env

# Use ".conda_env" as the name of the environment instead of the full path
conda config --set env_prompt '({name})'

# Activate the environment
conda activate $(pwd)/.bh_env

    

### GET DATA ###
mkdir data
mkdir data/raw

# Download data from from URL
curl -O https://biosampleplus.s3.ap-northeast-1.amazonaws.com/releases/2026-06_mistral-small3.1-24b.tar.gz
tar xzf 2026-06_mistral-small3.1-24b.tar.gz -C data
