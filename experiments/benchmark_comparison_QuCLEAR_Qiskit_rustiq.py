import re
import qiskit
import json
import time
import sys
import os
import argparse
import numpy as np
from qiskit import transpile, QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeManhattanV2

# Dynamically add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(os.getcwd())))  # Adjust path to root
sys.path.append(project_root)

from src.CE_module import construct_qcc_circuit, CE_recur_tree
from src.utilities import load_sycamore_coupling_map
from benchmarks.UCCSD_entanglers import generate_UCCSD_entanglers

from rustiq import pauli_network_synthesis, Metric
from rustiq.utils import entangling_count, entangling_depth

def convert_to_circuit(circuit: list) -> QuantumCircuit:
    """
    Converts a circuit into a CNOT + H + S + RZ(theta) to a quantumcircuit
    """
    num_qubits=len(set([num for op in circuit for num in op[1]]))
    new_circuit=QuantumCircuit(num_qubits)
    for gate in circuit:
        op=gate[0]
        params=gate[1]
        assert op in ["CNOT", "CZ", "H", "S", "Sd", "X", "Z", "SqrtX", "SqrtXd"], f"op not allowed {op}"
        if op=="CNOT":
            new_circuit.cx(params[0], params[1])
        elif op=="CZ":
            new_circuit.cz(params[0], params[1])
        elif op=="H":
            new_circuit.h(params[0])
        elif op=="S":
            new_circuit.s(params[0])
        elif op=="Sd":
            new_circuit.sdg(params[0])
        elif op=="X":
            new_circuit.x(params[0])
        elif op=="Z":
            new_circuit.z(params[0])
        elif op=="SqrtX":
            new_circuit.sx(params[0])
        elif op=="SqrtXd":
            new_circuit.sxdg(params[0])
    return new_circuit


#Compare json files in a specific folder
def run_experiment_folder(folder_path = None, filename = None, save_output = False, threshold = 1):

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
                paulis = json.load(file)

            # Function to check if a string contains only 'I' characters
            def is_all_identity(pauli):
                return all(char == 'I' for char in pauli)

            # Filter the list to remove all identity Paulis
            test_paulis = [pauli for pauli in paulis if not is_all_identity(pauli)]
            test_params = [0.01 * i for i in range(len(test_paulis))]

            # Measure time for our method
            start_time = time.time()
            opt_qc_f, append_clifford_f, sorted_entanglers_f = CE_recur_tree(entanglers=test_paulis, params=test_params, barrier=False, threshold = threshold)
            end_time = time.time()
            our_time = end_time - start_time

            start_time = time.time()
            opt_qc_f2, append_clifford_f2, sorted_entanglers_f2 = CE_recur_tree(entanglers=test_paulis, params=test_params, barrier=False, threshold = threshold)
            opt_qiskit = transpile(opt_qc_f2, optimization_level=3, basis_gates=["cx", "sx", "x", "rz"])
            end_time = time.time()
            combined_time = end_time - start_time
            
            # Measure time for Qiskit method
            start_time = time.time()
            origin_qc = construct_qcc_circuit(entanglers=test_paulis, params=test_params, barrier=False)
            origin_qiskit = transpile(origin_qc, optimization_level=3, basis_gates=["cx", "sx", "x", "rz"])
            end_time = time.time()
            qiskit_time = end_time - start_time

            # Measure time for RustiQ method
            start_time = time.time()
            circuit = pauli_network_synthesis(test_paulis, Metric.COUNT, True, fix_clifford=True)
            end_time = time.time()
            rustiq_time = end_time - start_time

            #Map to physical devices:
            opt_combine_ibm = transpile(opt_qc_f2, optimization_level=3, basis_gates=["cx", "sx", "x", "rz"], coupling_map=FakeManhattanV2().coupling_map)
            opt_combine_google = transpile(opt_qc_f2, optimization_level=3, basis_gates=["cx", "sx", "x", "rz"], coupling_map=load_sycamore_coupling_map())
            origin_qiskit_ibm = transpile(origin_qc, optimization_level=3, basis_gates=["cx", "sx", "x", "rz"], coupling_map=FakeManhattanV2().coupling_map)
            origin_qiskit_google = transpile(origin_qc, optimization_level=3, basis_gates=["cx", "sx", "x", "rz"], coupling_map=load_sycamore_coupling_map())
            
            # qiskit_circ_rustiq=convert_to_circuit(circuit)
            # rustiq_ibm = transpile(qiskit_circ_rustiq, optimization_level=3, basis_gates=["cx", "sx", "x", "rz"], coupling_map=FakeManhattanV2().coupling_map)
            # rustiq_google = transpile(qiskit_circ_rustiq, optimization_level=3, basis_gates=["cx", "sx", "x", "rz"], coupling_map=load_sycamore_coupling_map())
          
            # Collect results
            result_fc = {
                "num_paulis": len(test_paulis),
                "times": {
                    "our_time": our_time,
                    "combined_time": combined_time,
                    "qiskit_time": qiskit_time,
                    "rustiq_time": rustiq_time
                },
                "gate_counts": {
                    "our_method": opt_qc_f.count_ops().get('cx', 0),
                    "combined_method": opt_qiskit.count_ops().get('cx', 0),
                    "qiskit_method": origin_qiskit.count_ops().get('cx', 0),
                    "rustiq_method": entangling_count(circuit)
                },
                "circuit_entangling_depth": {
                    "our_method": opt_qc_f.depth(lambda instr: len(instr.qubits) > 1),
                    "combined_method": opt_qiskit.depth(lambda instr: len(instr.qubits) > 1),
                    "qiskit_method": origin_qiskit.depth(lambda instr: len(instr.qubits) > 1),
                    "rustiq_method": entangling_depth(circuit)
                },
                "test_paulis_file": f'experiments/results_fullyconnected/test_quclear_' + filename
            }
            print(result_fc)
            if save_output == True:
                # Save test_paulis to a separate JSON file
                with open(f'../experiments/results_fullyconnected/test_quclear_' + filename, 'w') as paulis_file:
                    json.dump([test_paulis, [result_fc]], paulis_file, indent=4)

            # Collect results for google mapping
            result_google = {
                "num_paulis": len(test_paulis),
                "gate_counts": {
                    "combined_method": opt_combine_google.count_ops().get('cx', 0),
                    "qiskit_method": origin_qiskit_google.count_ops().get('cx', 0),
                },
                "circuit_entangling_depth": {
                    "combined_method": opt_combine_google.depth(lambda instr: len(instr.qubits) > 1),
                    "qiskit_method": origin_qiskit_google.depth(lambda instr: len(instr.qubits) > 1),
                },
                "test_paulis_file": f'experiments/results_google/test_quclear_' + filename
            }
            print(result_google)
            if save_output == True:
                # Save test_paulis to a separate JSON file
                with open(f'../experiments/results_google/test_quclear_' + filename, 'w') as paulis_file:
                    json.dump([test_paulis, [result_google]], paulis_file, indent=4)
    
            # Collect results for ibm mapping
            result_ibm = {
                "num_paulis": len(test_paulis),
                "gate_counts": {
                    "combined_method": opt_combine_ibm.count_ops().get('cx', 0),
                    "qiskit_method": origin_qiskit_ibm.count_ops().get('cx', 0),
                },
                "circuit_entangling_depth": {
                    "combined_method": opt_combine_ibm.depth(lambda instr: len(instr.qubits) > 1),
                    "qiskit_method": origin_qiskit_ibm.depth(lambda instr: len(instr.qubits) > 1),
                },
                "test_paulis_file": f'experiments/results_ibm/test_quclear_' + filename
            }
            print(result_ibm)
            if save_output == True:
                # Save test_paulis to a separate JSON file
                with open(f'../experiments/results_ibm/test_quclear_' + filename, 'w') as paulis_file:
                    json.dump([test_paulis, [result_ibm]], paulis_file, indent=4)

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
            "../benchmarks/uccsd_paulis",
            "../benchmarks/HS_paulis",
            "../benchmarks/max_cut_paulis",
            "../benchmarks/labs_paulis"
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
        run_experiment_folder(folder_path=folder, save_output=True, threshold=1)

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


    