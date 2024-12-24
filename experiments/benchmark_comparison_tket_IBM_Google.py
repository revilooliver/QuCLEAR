import json
import time
import os
import sys
import argparse
from pytket.passes import FullPeepholeOptimise, KAKDecomposition
from pytket import Circuit, Qubit
from pytket.circuit import PauliExpBox, OpType, fresh_symbol
from pytket.pauli import Pauli, QubitPauliString
from pytket.passes import PauliSimp, DecomposeSingleQubitsTK1
from pytket.extensions.qiskit import tk_to_qiskit
from pytket.passes import RoutingPass, DecomposeSwapsToCXs
from pytket.architecture import Architecture
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeManhattanV2
from qiskit.transpiler import CouplingMap
current_dir = os.getcwd()
# Dynamically add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(os.getcwd())))  # Adjust path to root
sys.path.append(project_root)
from src.utilities import load_sycamore_coupling_map

# Pytket
pauli_dict={"I": Pauli.I, "X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}
def pytket_run(parr, coupling_map):
    all_symbols=[fresh_symbol(f"s{x}") for x in range(len(parr))]
    n = len(parr[0][0])
    q = [Qubit(i) for i in range(n)]
    def to_pauli_list(ps):
        r = []
        for i in ps:
            r.append(pauli_dict[i])
        return r
    # for i in parr:
    all_qubit_pauli_strings=[[QubitPauliString(q,to_pauli_list(p)) for p in block] for block in parr]

    def add_excitation(circ, all_ops, param=1.0):
        for term, symbol in zip(all_ops, all_symbols):
            # each term is one block so they should have the same parameter symbol.
            for pauli_str in term:
                qubits, paulis = zip(*pauli_str.to_list())
                paulis=[pauli_dict[p] for p in paulis]
                pbox = PauliExpBox(paulis, symbol * param)
                # transform the string form of the qubits to qubits
                qubits=[circ.get_q_register(q_list[0])[q_list[1][0]] for q_list in qubits]
                circ.add_pauliexpbox(pbox, qubits)
    ansatz = Circuit(n)
    
    # print(type(ansatz))
    add_excitation(ansatz, all_qubit_pauli_strings)
    # render_circuit_jupyter(ansatz)
    PauliSimp().apply(ansatz)
    # DecomposeBoxes().apply(ansatz)
    FullPeepholeOptimise().apply(ansatz) #heavy optimization
    DecomposeSingleQubitsTK1().apply(ansatz)

    coup_map=coupling_map#FakeSherbrooke().coupling_map
    edges=[e for e in coup_map.get_edges()]
    arch=Architecture(edges)
    RoutingPass(arch).apply(ansatz)
    DecomposeSwapsToCXs(arch).apply(ansatz)
    # KAKDecomposition().apply(ansatz)
    # render_circuit_jupyter(ansatz)
    print(f"CNOT: {ansatz.n_gates_of_type(OpType.CX)}, Single: {ansatz.n_gates-ansatz.n_gates_of_type(OpType.CX)}, Total: {ansatz.n_gates}, Depth: {ansatz.depth()}")

    return ansatz


def is_all_identity(pauli):
    return all(char == 'I' for char in pauli)

#Compare json files in a specific folder
def run_experiment_folder(folder_path = None, filename = None, save_output = False, device = 'IBM'):

    if filename == None:
        file_list = os.listdir(folder_path)
    else:
        file_list = [filename]
    if device == 'IBM':
        coupling_map = FakeManhattanV2().coupling_map
        save_path = "results_ibm"
    elif device == 'Google':
        coupling_map = CouplingMap(load_sycamore_coupling_map())
        save_path = "results_google"

    # Iterate over all files in the folder
    for filename in file_list:
        # Check if the file is a JSON file
        if filename.endswith(".json"):
            results = []
            # Print the filename
            print(filename)
    
            with open(folder_path + '/' + filename, "r") as file:
                test_paulis = json.load(file)
            # Filter the list to remove all identity Paulis
            if filename.startswith(("max_cut", "labs")): # qaoa requires a block structure
                paulis=[[p for p in block if not is_all_identity(p)] for block in test_paulis]
                paulis=[block for block in paulis if len(block)!=0]
                number_of_ham=len([p for block in paulis for p in block])
            else: # not qaoa so not block structure
                paulis = [[p] for p in test_paulis if not is_all_identity(p)]
                number_of_ham=len(paulis)

            # Measure time for Tetris method
            start_time = time.time()
            circuit = pytket_run(paulis, coupling_map = coupling_map)
            end_time = time.time()
            circuit_qiskit=tk_to_qiskit(circuit)
            tot_time = end_time - start_time
        
            # Collect results
            result = {
                "num_paulis": number_of_ham,
                "times": {
                    "pytket_time": tot_time
                },
                "gate_counts": {
                    "pytket_method": circuit.n_gates_of_type(OpType.CX)
                },
                "circuit_entangling_depth": {
                    "pytket_method": circuit_qiskit.depth(lambda instr: len(instr.qubits) > 1)
                },
                "test_paulis_file": f'experiments/' + save_path + '/test_pytket_' + filename
            }
            print(result)
            results.append(result)
            if save_output == True:
                # Save test_paulis to a separate JSON file
                with open(f'../experiments/' + save_path +'/test_pytket_' + filename, 'w') as paulis_file:
                    json.dump([test_paulis, results], paulis_file, indent=4)
    

    
#Compare a given list of paulis
def run_experiment_paulis(test_paulis, test_params = None, save_output = False, filename = "Paulis", device = 'IBM'):

    results = []
    # Filter the list to remove all identity Paulis
    if test_params is None:
        test_params = [0.01 * i for i in range(len(test_paulis))]
    
    number_of_ham=len([p for block in test_paulis for p in block])
    if device == 'IBM':
        coupling_map = FakeManhattanV2().coupling_map
        save_path = "results_ibm"
    elif device == 'Google':
        coupling_map = CouplingMap(load_sycamore_coupling_map())
        save_path = "results_google"
    # Measure time for pytket method
    start_time = time.time()
    circuit = pytket_run(test_paulis, coupling_map = coupling_map)
    end_time = time.time()
    circuit_qiskit=tk_to_qiskit(circuit)
    tot_time = end_time - start_time

    # Collect results
    result = {
        "num_paulis": number_of_ham,
        "times": {
            "pytket_time": tot_time
        },
        "gate_counts": {
            "pytket_method": circuit.n_gates_of_type(OpType.CX)
        },
        "circuit_entangling_depth": {
            "pytket_method": circuit_qiskit.depth(lambda instr: len(instr.qubits) > 1)
        },
        "test_paulis_file": f'experiments/' + save_path + '/test_pytket_' + filename
    }
    print(result)
    results.append(result)
    if save_output == True:
        # Save test_paulis to a separate JSON file
        with open(f'../experiments/' + save_path + '/test_pytket_' + filename, 'w') as paulis_file:
            json.dump([test_paulis, results], paulis_file, indent=4)
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
            "../benchmarks/hardware_mapping_blocks",
            # "../benchmarks/max_cut_paulis_blocks",
            # "../benchmarks/labs_paulis_blocks"
        ]
    }
    
    # Validate configuration
    if config not in benchmark_paths:
        print(f"Error: Invalid configuration '{config}'. Use 'fast' or 'full'.")
        sys.exit(1)
    
    # Run experiments
    print(f"Running benchmarks in '{config}' mode for IBM device connectivity...")
    for folder in benchmark_paths[config]:
        print(f"Running benchmarks in folder: {folder}")
        run_experiment_folder(folder_path=folder, save_output=True, device = 'IBM')

    print(f"Running benchmarks in '{config}' mode for Google device connectivity...")
    for folder in benchmark_paths[config]:
        print(f"Running benchmarks in folder: {folder}")
        run_experiment_folder(folder_path=folder, save_output=True, device = 'Google')
        
    if config == 'full':
        electrons_list = [10]
        orbitals_list = [20]
        #First evaluate the UCCSD ansatz:
        for e, o in zip(electrons_list, orbitals_list):
            filename=f"uccsd_paulis_e{e}_o{o}.json"
            with open(f"../benchmarks/uccsd_paulis_blocks/{filename}", "r") as file:
                data=json.load(file)
            data = [[p for p in block if not is_all_identity(p)] for block in data]
            data= [block for block in data if len(block)!=0]
            num_hams=len([p for block in data for p in block])
            entanglers = run_experiment_paulis(data, save_output = True, filename=f"Paulis{num_hams}.json", device = 'IBM')
            entanglers = run_experiment_paulis(data, save_output = True, filename=f"Paulis{num_hams}.json", device = 'Google')


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