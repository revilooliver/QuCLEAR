#!/bin/bash

# Exit immediately on error
set -e

# Initialize conda for non-interactive shell
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found. Ensure Conda is properly installed."
    exit 1
fi

# Activate Conda environment
ENV_NAME="PH_env"
echo "Activating Conda environment: $ENV_NAME"
if conda activate "$ENV_NAME"; then
    echo "Successfully activated $ENV_NAME."
else
    echo "Error: Failed to activate Conda environment '$ENV_NAME'."
    exit 1
fi

# Run benchmark scripts
echo "Running PH init scripts..."
for script in "benchmark_comparison_PH_init.py"; do
    if [ -f "$script" ]; then
        echo "Executing: python $script fast"
        python "$script" fast
        echo "Successfully executed $script."
    else
        echo "Error: Script '$script' not found in $SCRIPT_DIR."
        exit 1
    fi
done

# Deactivate environment
echo "Deactivating Conda environment..."
conda deactivate
echo "Environment deactivated."

# Activate Conda environment
ENV_NAME="QuCLEAR_env"
echo "Activating Conda environment: $ENV_NAME"
if conda activate "$ENV_NAME"; then
    echo "Successfully activated $ENV_NAME."
else
    echo "Error: Failed to activate Conda environment '$ENV_NAME'."
    exit 1
fi

# Run benchmark scripts
echo "Running PH init scripts..."
for script in "benchmark_comparison_PH_opt.py"; do
    if [ -f "$script" ]; then
        echo "Executing: python $script fast"
        python "$script" fast
        echo "Successfully executed $script."
    else
        echo "Error: Script '$script' not found in $SCRIPT_DIR."
        exit 1
    fi
done

# Deactivate environment
echo "Deactivating Conda environment..."
conda deactivate
echo "Environment deactivated."

echo "The results are stored in results_fullyconnected folder in experiments folder"
echo "All tasks completed successfully."