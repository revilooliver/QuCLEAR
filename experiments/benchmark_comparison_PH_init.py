# This file needs to be executed in the PH_env since it contains the older Qiskit version compatible with PH to initialize

import time
import json
import os
import sys
import argparse
import pickle
# Save the current working directory
current_dir = os.getcwd()

# Move to the target directory
target_dir = os.path.abspath("../Paulihedral_new")
os.chdir(target_dir)

# Add target directory to Python's module search path
if target_dir not in sys.path:
    sys.path.insert(0, target_dir)
from benchmark.mypauli import pauliString
import synthesis_FT, parallel_bl
import importlib
importlib.reload(synthesis_FT)
importlib.reload(parallel_bl)
os.chdir(current_dir)
from copy import deepcopy

def paulihedral(parr):
    # nq = len(parr[0][0])
    # length = nq//2 # `length' is a hyperparameter, and can be adjusted for best performance
    # a1 = parallel_bl.depth_oriented_scheduling(parr, length=length, maxiter=30)
    a1 = parallel_bl.gate_count_oriented_scheduling(parr)
    a1=[[[p]] for block1 in a1 for block2 in block1 for p in block2] #flatten
    qc = synthesis_FT.block_opt_FT(a1)
    # print(qc)

    return qc

def is_all_identity(pauli):
    return all(char == 'I' for char in pauli)


#Compare json files in a specific folder
def run_experiment_folder(folder_path = None, filename = None, save_output = False):

    if filename == None:
        file_list = os.listdir(folder_path)
    else:
        file_list = [filename]
    # Iterate over all files in the folder
    for filename in file_list:
        # Check if the file is a JSON file
        if filename.endswith(".json"):
            results = []
            # Print the filename
            print(filename)
            with open(folder_path + '/' + filename, "r") as file:
                test_paulis = json.load(file)
                origin_paulis=deepcopy(test_paulis) # we will save this.
            if filename.startswith(("max_cut", "labs")):
                test_paulis = test_paulis[0] # only get the cost hamiltonian
            number_of_ham=len(test_paulis)
            test_params = [0.01 * i for i in range(len(test_paulis))]

            # Measure time for method
            start_time = time.time()
            # Filter the list to remove all identity Paulis
            paulis = [[pauliString(pauli, 1.0)] for pauli in test_paulis if not is_all_identity(pauli)]
            circuit = paulihedral(paulis)
            end_time = time.time()
            tot_time = end_time - start_time
        
            # Collect results
            result = {
                "num_paulis": number_of_ham,
                "times": {
                    "paulihedral_time": tot_time
                },
                # "gate_counts": {
                #     "paulihedral_method": circuit.count_ops().get('cx', 0)
                # },
                # "circuit_depth": {
                #     "paulihedral_method": circuit.depth()
                # },
                # "test_paulis_file": f'benchmarks/results/paulihedral_partial_results/' + filename
            }
            print(result)
            results.append(result)
            if save_output == True:
                # Save to pickle files
                save_filename=filename[0:-len(".json")]+".pkl"
                with open(f'../experiments/paulihedral_partial_results/' + save_filename, 'wb') as paulis_file:
                    pickle.dump({"paulis": origin_paulis, "circuit": circuit.qasm(), "results": results}, paulis_file)
                # with open(f'benchmarks/paulihedral_partial_results/' + save_filename, 'rb') as paulis_file:
                #     print("printing data: ", pickle.load(paulis_file))
    

    #Compare a given list of paulis
def run_experiment_paulis(test_paulis, test_params = None, save_output = False, filename = "Paulis"):

    results = []
    # paulis = test_paulis
    # Filter the list to remove all identity Paulis
    if test_params is None:
        test_params = [0.01 * i for i in range(len(test_paulis))]
    
    number_of_ham=len([p for block in test_paulis for p in block])
    # Measure time for method
    start_time = time.time()
    paulis = [[pauliString(p, 1.0) for p in block] for block in test_paulis]
    # paulis = [[pauliString(p, 1.0)] for p in test_paulis]
    circuit= paulihedral(paulis)
    # circuit = transpile(circuit, basis_gates=["cx", "sx", "x", "rz"], optimization_level=3)
    # print(circuit)
    end_time = time.time()
    tot_time = end_time - start_time

    # Collect results
    result = {
        "num_paulis": number_of_ham,
        "times": {
            "paulihedral_time": tot_time
        }
        # "gate_counts": {
        #     "paulihedral_method": circuit.count_ops().get('cx', 0)
        # },
        # "circuit_depth": {
        #     "paulihedral_method": circuit.depth()
        # },
        # "test_paulis_file": f'benchmarks/paulihedral_partial_results' + filename
    }
    print(result)
    results.append(result)
    if save_output == True:
        # Save to pickle files
        with open(f'../experiments/paulihedral_partial_results/' + filename, 'wb') as paulis_file:
            pickle.dump({"paulis": test_paulis, "circuit": circuit.qasm(), "results": results}, paulis_file)
        # with open(f'benchmarks/paulihedral_partial_results/' + filename, 'rb') as paulis_file:
        #     print("printing data: ", pickle.load(paulis_file))
    return circuit
    

def run_benchmarks(config):
    """
    Main function to compile and run benchmarks based on the specified configuration.

    Args:
        config (str): The benchmark configuration mode. 
                      "fast" runs a minimal set of benchmarks.
                      "full" runs all available benchmarks.
    """
    # Define benchmark folders
    benchmark_paths = {
        "fast": ["../benchmarks/HS_paulis"],
        "full": [
            "../benchmarks/HS_paulis",
            "../benchmarks/max_cut_paulis_blocks",
            "../benchmarks/labs_paulis_blocks"
        ]
    }
    
    # Validate configuration
    if config not in benchmark_paths:
        print(f"Error: Invalid configuration '{config}'. Use 'fast' or 'full'.")
        sys.exit(1)
    
    # Run experiments
    print(f"Running benchmarks in '{config}' mode...")
    for folder in benchmark_paths[config]:
        print(f"Running benchmarks in folder: {folder}")
        run_experiment_folder(folder_path=folder, save_output=True)
        
    if config == 'full':
        #first compare the UCCSD ansatz
        electrons_list = [2, 2, 4, 6, 8, 10,]
        orbitals_list = [4, 6, 8, 12, 16, 20]
        # paulis_len=[24, 80, 320, 1656, 5376, 13400]
        #First evaluate the UCCSD ansatz:
        for e, o in zip(electrons_list, orbitals_list):
            filename=f"uccsd_hamiltonian_e{e}_o{o}.json"
            with open(f"../benchmarks/uccsd_hamiltonians/{filename}", "r") as file:
                data=json.load(file)
            num_qubits=len(data[0][0])
            
            # data=[[p for p in block if not is_all_identity(p)] for block in data]
            # data=[block for block in data if len(block)>0]
            # data=[p for block in data for p in block if not is_all_identity(p)] #flatten
            num_hams=len([p for block in data for p in block])
            entanglers = run_experiment_paulis(data, num_qubits, save_output = True, filename=f"Paulis{num_hams}.pkl")


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