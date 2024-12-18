#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to check if a file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo "Error: File '$1' not found. Exiting."
        exit 1
    fi
}

# Function to check if a Conda environment exists
env_exists() {
    conda env list | grep -qE "^$1\s"
}

# Function to create and activate a Conda environment
create_and_activate_env() {
    local env_file=$1
    local env_name=$2

    echo "----------------------------------------"
    echo "Checking for existing Conda environment: $env_name..."

    if env_exists "$env_name"; then
        echo "Environment '$env_name' already exists. Skipping creation."
    else
        echo "Creating Conda environment '$env_name'..."
        check_file "$env_file"

        conda env create -f "$env_file" || { echo "Failed to create $env_name environment"; exit 1; }
        echo "Environment '$env_name' created successfully."
    fi

    echo "Activating $env_name environment..."
    # Shell-compatible activation
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$env_name"
    echo "Activated $env_name environment."
}

# QuCLEAR Installation
echo "========== Running QuCLEAR Installation =========="
create_and_activate_env "QuCLEAR_env.yaml" "QuCLEAR_env"

echo "Installing QuCLEAR dependencies..."
cd ..
pip install --no-cache-dir -r requirements.txt
echo "QuCLEAR dependencies installed successfully."
cd artifact_evaluation

echo "Deactivating QuCLEAR environment..."
conda deactivate

# PH Installation
echo "========== Running PH Installation =========="
create_and_activate_env "PH_env.yaml" "PH_env"

echo "Installing PH dependencies..."
cd ../Paulihedral_new
pip install --no-cache-dir -r requirements.txt
echo "PH dependencies installed successfully."

echo "Deactivating PH environment..."
cd ../artifact_evaluation
conda deactivate

echo "----------------------------------------"
echo "QuCLEAR, Pytket, and PH installations completed successfully. Now please install RustiQ."
