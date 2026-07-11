#!/usr/bin/env bash

# Setting up error management during installation.
#
#   - A command exits with a non-zero status ``-e``
#   - An undefined variable is being used ``-u``
#   - A pipeline fails anywhere ``-o pipefail``
#
set -euo pipefail

# name of the virtual environment as well as define python version
VENV=".venv"
PYTHON="${1:-python3.11}"

"$PYTHON" -m venv "$VENV"

# Activates the environment
#   -- Shellcheck disable=SC1090
source "$VENV/bin/activate"

# ----------------------------------------
# Core Tooling

python -m pip install --upgrade pip setuptools wheel


# ----------------------------------------
# 3rd party Libraries 

pip install                         \
    "tensorflow[and-cuda]==2.20.0"  \
    pandas                          \
    "numpy<2.0"                     \
    scipy                           \
    pyarrow                         \
    numba                           \
    matplotlib                      \
    seaborn                         \
    tqdm                            \
    orjson                          \
    einops                          \
    h5py                            \
    plotly                          \
    tensorboard


pip install git+https://github.com/NVlabs/sionna.git@main

pip freeze > requirements.txt

