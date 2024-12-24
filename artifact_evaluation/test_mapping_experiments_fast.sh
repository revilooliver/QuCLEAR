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
ENV_NAME="QuCLEAR_env"
echo "Activating Conda environment: $ENV_NAME"
if conda activate "$ENV_NAME"; then
    echo "Successfully activated $ENV_NAME."
else
    echo "Error: Failed to activate Conda environment '$ENV_NAME'."
    exit 1
fi

# Change to experiments directory
SCRIPT_DIR="../experiments"
if [ -d "$SCRIPT_DIR" ]; then
    cd "$SCRIPT_DIR"
    echo "Changed directory to $SCRIPT_DIR."
else
    echo "Error: Directory '$SCRIPT_DIR' does not exist."
    exit 1
fi

# Run benchmark scripts
echo "Running benchmark scripts..."
for script in "benchmark_comparison_tket_IBM_Google.py"; do
    if [ -f "$script" ]; then
        echo "+++++++++++++++++++++++++++++++++++++++++Running $script evaluation+++++++++++++++++++++++++++++++++++++++++"
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

echo "+++++++++++++++++++++++++++++++++++++++++Running Paulihedral evaluation+++++++++++++++++++++++++++++++++++++++++"

./compare_paulihedral_hardware_fast.sh
echo "+++++++++++++++++++++++++++++++++++++++++Running Tetris evaluation+++++++++++++++++++++++++++++++++++++++++"
./compare_tetris_hardware_fast.sh
echo "Paulihedral test success"

echo "The results are stored in results_ibm, and resutls_google folder in experiments folder"
echo "All tasks completed successfully."
