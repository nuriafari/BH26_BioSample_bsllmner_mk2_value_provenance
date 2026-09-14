# #!/bin/bash

# Activate environment
conda config --set env_prompt '({name})'
conda activate $(pwd)/.conda_env

# To update environment
# conda env update -f environment.yaml --prefix .conda_env
curl -fsSL https://claude.ai/install.sh | bash -s stable

# Download Claude with CURL
export PATH="$HOME/.local/bin:$PATH"
claude
