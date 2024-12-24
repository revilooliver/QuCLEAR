# This file needs to be executed in the PH_env since it contains the older Qiskit version compatible with PH to initialize

import time
import json
import os
import sys
import argparse
import pickle
import importlib
from copy import deepcopy
# Save the current working directory
current_dir = os.getcwd()
# Dynamically add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(os.getcwd())))  # Adjust path to root
sys.path.append(project_root)
from src.utilities import load_sycamore_coupling_map
# Move to the target directory
target_dir = os.path.abspath("../vqe_tetris-master/core")
os.chdir(target_dir)

# Add target directory to Python's module search path
if target_dir not in sys.path:
    sys.path.insert(0, target_dir)
from utils import synthesis_lookahead, hardware, synthesis_max_cancel
from utils.hardware import graph_from_coupling
from benchmark.mypauli import pauliString
# importlib.reload(synthesis_lookahead)
from qiskit.transpiler import CouplingMap
# from tools import print_qc
os.chdir(current_dir)

def Tetris_lookahead(parr, use_bridge=False, swap_coefficient=3, k=10):

    if k>len(parr): #avoids exceeding the index limit
        k=len(parr)-1
    graph=graph_from_coupling(load_sycamore_coupling_map())
    # k=10
    qc, metrics = synthesis_lookahead.synthesis_lookahead(parr, graph=graph, arch='manhattan', 
                    use_bridge=use_bridge, swap_coefficient=swap_coefficient, k=k) #arch doesn't do anything since we pass a graph.

    return qc

def is_all_identity(pauli):
    return all(char == 'I' for char in pauli)

#Compare json files in a specific folder
import os, json
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
            circuit = Tetris_lookahead(paulis)
            # circuit = transpile(circuit, basis_gates=["cx", "sx", "x", "rz"], optimization_level=3)
            end_time = time.time()
            tot_time = end_time - start_time
        
            # Collect results
            result = {
                "num_paulis": number_of_ham,
                "times": {
                    "tetris_time": tot_time
                },
                # "gate_counts": {
                #     "tetris_method": circuit.count_ops().get('cx', 0)
                # },
                # "circuit_depth": {
                #     "tetris_method": circuit.depth()
                # },
                # "test_paulis_file": f'benchmarks/results/tetris_partial_results/' + filename
            }
            results.append(result)
            if save_output == True:
                # Save to pickle files
                save_filename=filename[0:-len(".json")]+".pkl"
                with open(f'../experiments/tetris_partial_results_google/' + save_filename, 'wb') as paulis_file:
                    pickle.dump({"paulis": origin_paulis, "circuit": circuit.qasm(), "results": results}, paulis_file)
                # with open(f'benchmarks/tetris_partial_results/' + save_filename, 'rb') as paulis_file:
                #     print("printing data: ", pickle.load(paulis_file))
    

#Compare a given list of paulis
def run_experiment_paulis(test_paulis, test_params = None, save_output = False, filename = "Paulis"):

    results = []
    # paulis = test_paulis
    # Filter the list to remove all identity Paulis
    if test_params is None:
        test_params = [0.01 * i for i in range(len(test_paulis))]
    
    number_of_ham=len([p for block in test_paulis for p in block if not is_all_identity(p)])
    # Measure time for Tetris method
    paulis = [[p for p in block if not is_all_identity(p)] for block in test_paulis]
    paulis=[block for block in paulis if block] #remove identities
    #process
    start_time = time.time()
    paulis = [[pauliString(p, 1.0) for p in block if not is_all_identity(p)] for block in test_paulis]
    paulis=[block for block in paulis if block]
    circuit_tetris = Tetris_lookahead(paulis)
    # circuit_tetris = transpile(circuit_tetris, basis_gates=["cx", "sx", "x", "rz"], optimization_level=3)
    end_time = time.time()
    tetris_time = end_time - start_time

    # Collect results
    result = {
        "num_paulis": number_of_ham,
        "times": {
            # "our_time": our_time,
            # "combined_time": combined_time,
            # "qiskit_time": qiskit_time,
            # "rustiq_time": rustiq_time
            "tetris_time": tetris_time
        }
        # "gate_counts": {
        #     # "our_method": opt_qc_f.count_ops().get('cx', 0),
        #     # "combined_method": opt_qiskit.count_ops().get('cx', 0),
        #     # "qiskit_method": origin_qiskit.count_ops().get('cx', 0),
        #     # "rustiq_method": entangling_count(circuit),
        #     "tetris_method": circuit_tetris.count_ops().get('cx', 0)
        # },
        # "circuit_entangling_depth": {
        #     # "our_method": opt_qc_f.depth(lambda instr: len(instr.qubits) > 1),
        #     # "combined_method": opt_qiskit.depth(lambda instr: len(instr.qubits) > 1),
        #     # "qiskit_method": origin_qiskit.depth(lambda instr: len(instr.qubits) > 1),
        #     # "rustiq_method": entangling_depth(circuit),
        #     "tetris_method": circuit_tetris.depth(lambda instr: len(instr.qubits) > 1)
        # },
        # "test_paulis_file": f'benchmarks/results/test_tetris_' + filename
    }
    results.append(result)
    if save_output == True:
        # Save to pickle files
        with open(f'../experiments/tetris_partial_results_google/' + filename, 'wb') as paulis_file:
            pickle.dump({"paulis": test_paulis, "circuit": circuit_tetris.qasm(), "results": results}, paulis_file)
        # with open(f'benchmarks/paulihedral_partial_results/' + filename, 'rb') as paulis_file:
        #     print("printing data: ", pickle.load(paulis_file))
    return circuit_tetris
    
    

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