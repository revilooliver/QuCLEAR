# Execute with lates qiskit to keep the time CX comparisons equal.
# This file needs to be executed in the QuCLEAR_env since it contains the latest Qiskit version to optimize

import time
import json
import os
import pickle
import sys
import argparse
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeManhattanV2
from copy import deepcopy

#Load the pickle files.
def run_experiment_folder(folder_path = None, filename = None, save_output = False, names_start_with=""):

    if filename == None:
        file_list = os.listdir(folder_path)
    else:
        file_list = [filename]
    # Iterate over all files in the folder
    for filename in file_list:
        # Check if the file is a JSON file
        if filename.endswith(".pkl") and filename.startswith(names_start_with):
            # Print the filename
            with open(folder_path + filename, "rb") as file:
                data = pickle.load(file)
            paulis=data["paulis"]
            circuit=QuantumCircuit.from_qasm_str(data["circuit"])
            results=data["results"][0]
            tot_time=results["times"]["paulihedral_time"]
            number_of_ham=results["num_paulis"]
            # Measure time for method
            start_time = time.time()
            circuit = transpile(circuit, basis_gates=["cx", "sx", "x", "rz"], coupling_map=FakeManhattanV2().coupling_map, optimization_level=3)
            end_time = time.time()
            # include the transpilation time.
            tot_time+= (end_time - start_time)
            results = []
            # Collect results
            new_filename=filename[0:-len(".pkl")]+".json"

            if filename.startswith(("max_cut", "labs")): # we will multiply the number of cnot by the layer count since
                # ph does not optimize across qaoa layers.
                layers=int(len(paulis)/2)
            else:
                layers=1
            print("found layers: ", layers)
            # print("original ", circuit)
            circuit=circuit.repeat(layers)
            circuit=circuit.decompose()
            # print("repeated ", circuit)
            result = {
                "num_paulis": number_of_ham,
                "times": {
                    "paulihedral_time": tot_time
                },
                "gate_counts": {
                    "paulihedral_method": circuit.count_ops().get('cx', 0)*layers
                },
                "circuit_entangling_depth": {
                    "paulihedral_method": circuit.depth(lambda instr: len(instr.qubits) > 1)
                },
                "test_paulis_file": f'../experiments/results_ibm/test_paulihedral_' + new_filename
            }
            print(result)
            results.append(result)
            if save_output == True:
                # Save test_paulis to a separate JSON file
                with open(f'../experiments/results_ibm/test_paulihedral_' + new_filename, 'w') as paulis_file:
                    json.dump([paulis, results], paulis_file, indent=4)
    
def run_benchmarks(config):
    """
    Main function to compile and run benchmarks based on the specified configuration.

    Args:
        config (str): The benchmark configuration mode. 
                      "fast" runs a minimal set of benchmarks.
                      "full" runs all available benchmarks.
    """
    # Define benchmark folders
    names_start_with = {
        "fast": ("LiH", "H2O", "benzene"),
        "full": ("LiH", "H2O", "benzene", 'Paulis', "max_cut", 'labs'),
    }
    
    # Validate configuration
    if config not in names_start_with:
        print(f"Error: Invalid configuration '{config}'. Use 'fast' or 'full'.")
        sys.exit(1)
    
    # Run experiments
    print(f"Running benchmarks in '{config}' mode...")
    for name in names_start_with[config]:
        print(f"Running benchmarks in folder: ../experiments/paulihedral_partial_results_ibm/")
        run_experiment_folder(folder_path="../experiments/paulihedral_partial_results_ibm/", save_output=True, names_start_with=name)

    print("Benchmark execution completed successfully.")

def main():
    """
    Parse command-line arguments and run the appropriate benchmarks.
    """
    # Use argparse for cleaner argument parsing
    parser = argparse.ArgumentParser(description="Run benchmark experiments based on the specified configuration.")
    parser.add_argument("config", choices=["fast", "full"], 
                        help="Benchmark configuration: 'fast' for minimal benchmarks or 'full' for all benchmarks.")
    args = parser.parse_args()

    # Run benchmarks
    run_benchmarks(args.config)

if __name__ == "__main__":
    main()