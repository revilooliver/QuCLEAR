import numpy as np
import re
import qiskit
from typing import List
from itertools import product

from qiskit import *
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

from pytket import Circuit
from pytket.extensions.quantinuum import QuantinuumBackend

from utils import pauli_strings_commute

from vqe_utils import read_from_file, MeasureCircuit, evaluation

from pauli_utils import ChecksFinder

from utils import convert_pauli_list


import logging

# Configure logging
# Configure module-level logger
logger = logging.getLogger(__name__)


def construct_qcc_circuit(entanglers: List[str], params: List[float], truncation=None, barrier=False):
    '''This function defines the exponential building block for hamtilbonian simulation. 
    
    Args:
        entanglers: list storing Pauli words for construction of qcc_circuit.
        params: parameters for the rotations in blocks
    Returns:
        qcc_circuit
    '''
    if truncation != None:
        if len(entanglers) > truncation:
            num_blocks = truncation
        else:
            num_blocks = len(entanglers)
    else:
        num_blocks = len(entanglers)
    
    num_qubits = len(entanglers[0])
    qcc_circuit = QuantumCircuit(num_qubits)
    for i in range(num_blocks):
        circuit = QuantumCircuit(num_qubits)
        key = entanglers[i]
        coupler_map = []
        # We first construct coupler_map according to the key.
        for j in range(num_qubits):
            if key[num_qubits-1-j] != 'I':
                coupler_map.append(j)
                
        # Then we construct the circuit.
        if len(coupler_map) == 1:
            # there is no CNOT gate.
            c = coupler_map[0]
            if key[num_qubits-1-c] == 'X':
                circuit.h(c)
                circuit.rz(params[i], c)
                circuit.h(c)
            elif key[num_qubits-1-c] == 'Y': 
                circuit.sdg(c)
                circuit.h(c)
                circuit.rz(params[i], c)
                circuit.h(c)
                circuit.s(c)
            elif key[num_qubits-1-c] == 'Z':
                circuit.rz(params[i], c)
                
            qcc_circuit.compose(circuit, inplace=True)
        else:
            # Here we would need CNOT gate.
            for j in coupler_map:
                if key[num_qubits-1-j] == 'X':
                    circuit.h(j)
                elif key[num_qubits-1-j] == 'Y':
                    circuit.sdg(j)
                    circuit.h(j)
                    
            for j in range(len(coupler_map) - 1):
                circuit.cx(coupler_map[j], coupler_map[j+1])
                
            param_gate = QuantumCircuit(num_qubits)
            param_gate.rz(params[i], coupler_map[-1])
            
            qcc_circuit.compose(circuit, inplace=True)
            qcc_circuit.compose(param_gate, inplace=True)
            qcc_circuit.compose(circuit.inverse(), inplace=True)
        if barrier is True:
            qcc_circuit.barrier()
    
    return qcc_circuit


def construct_extracted_subcircuit(entangler: str, param: float):
    '''This function defines the extracted building block for hamtilbonian simulation. 
    
    Args:
        entangler: Pauli words for construction of optimized building block
        param: parameter associated with the building block
    Returns:
        qcc_subcircuit
    '''
    
    num_qubits = len(entangler)
    circuit = QuantumCircuit(num_qubits)
    key = entangler
    coupler_map = []
    # We first construct coupler_map according to the key.
    for j in range(num_qubits):
        if key[num_qubits-1-j] != 'I':
            coupler_map.append(j)
            
    # Then we construct the circuit.
    if len(coupler_map) == 1:
        # there is no CNOT gate.
        c = coupler_map[0]
        if key[num_qubits-1-c] == 'X':
            circuit.h(c)
            circuit.rz(param, c)
            # circuit.h(c)
        elif key[num_qubits-1-c] == 'Y':
            circuit.sdg(c)
            circuit.h(c)
            circuit.rz(param, c)
            # circuit.h(c)
            # circuit.s(c)
        elif key[num_qubits-1-c] == 'Z':
            circuit.rz(param, c)
    else:
        # Here we would need CNOT gate.
        for j in coupler_map:
            if key[num_qubits-1-j] == 'X':
                circuit.h(j)
            elif key[num_qubits-1-j] == 'Y':
                circuit.sdg(j)
                circuit.h(j)
                
        for j in range(len(coupler_map) - 1):
            circuit.cx(coupler_map[j], coupler_map[j+1])
            
        param_gate = QuantumCircuit(num_qubits)
        param_gate.rz(param, coupler_map[-1])
        
        # circuit.compose(circuit, inplace=True)
        circuit.compose(param_gate, inplace=True)
    
    return circuit

def construct_Clifford_subcircuit(entangler: str):
    '''This function defines the clifford subcircuit for hamtilbonian simulation. 
    
    Args:
        entangler: Pauli words for construction of optimized building block
    Returns:
        clifford_subcircuit
    '''
    
    num_qubits = len(entangler)
    circuit = QuantumCircuit(num_qubits)
    key = entangler
    coupler_map = []
    # We first construct coupler_map according to the key.
    for j in range(num_qubits):
        if key[num_qubits-1-j] != 'I':
            coupler_map.append(j)
            
    # Then we construct the circuit.
    if len(coupler_map) == 1:
        # there is no CNOT gate.
        c = coupler_map[0]
        if key[num_qubits-1-c] == 'X':
            circuit.h(c)
        elif key[num_qubits-1-c] == 'Y':
            circuit.h(c)
            circuit.s(c)
    else:
        # Here we would need CNOT gate.
        for j in range(len(coupler_map)-2, -1, -1):
            # print(len(coupler_map), j, j+1)
            circuit.cx(coupler_map[j], coupler_map[j+1])
            
        for j in coupler_map:
            if key[num_qubits-1-j] == 'X':
                circuit.h(j)
            elif key[num_qubits-1-j] == 'Y':
                circuit.h(j)
                circuit.s(j)
    # clifford_subcircuit = circuit.inverse()
    
    return circuit

def construct_sq_subcircuit(entangler: str):
    '''This function defines the single qubit clifford subcircuit for hamtilbonian simulation. 
    
    Args:
        entangler: Pauli words for construction of optimized building block
    Returns:
        clifford_subcircuit
    '''
    
    num_qubits = len(entangler)
    circuit = QuantumCircuit(num_qubits)
    key = entangler
    # We first construct coupler_map according to the key.
    for j in range(num_qubits):
        if key[num_qubits - 1 - j] == 'X':
            circuit.h(j)
        elif key[num_qubits - 1 - j] == 'Y':
            circuit.h(j)
            circuit.s(j)
    return circuit

def construct_opt_subcircuit(entangler: str, param: float, clifford_circuit: QuantumCircuit):
    '''This function calcu;ates the optimized subcircuit after passing the clifford circuit through the entangler.
    
    Args:
        entangler: Pauli words for construction of optimized building block
    Returns:
        clifford_optcircuit
    '''
    
    num_qubits = len(entangler)
    pauli_finder = ChecksFinder(num_qubits, clifford_circuit)
    pauli_result = pauli_finder.find_checks_sym(pauli_group_elem = entangler)
    opt_sign = pauli_result.p1_str[0:2]
    opt_pauli = pauli_result.p1_str[2:]
    # print(opt_sign, opt_pauli)
    new_param = 0
    if opt_sign == "+1":
        new_param = param
    elif opt_sign == "-1":
        new_param = -param
    else:
        raise Exception("Incorrect sign")

    extracted_qc = construct_extracted_subcircuit(opt_pauli, new_param)

    return pauli_result.p1_str, extracted_qc

def construct_opt_pauli(entangler: str, clifford_circuit: QuantumCircuit):
    '''This function calculates the optimized Pauli after passing the clifford circuit through the entangler.
    
    Args:
        entangler: Pauli words for construction of optimized building block
    Returns:
        clifford_optcircuit
    '''
    
    num_qubits = len(entangler)
    pauli_finder = ChecksFinder(num_qubits, clifford_circuit)
    pauli_result = pauli_finder.find_checks_sym(pauli_group_elem = entangler)

    return pauli_result.p1_str

def generate_opt_circuit(entanglers: List[str], params: List[float], barrier=False):
    '''This function defines the exponential building block for hamtilbonian simulation. 
    
    Args:
        entanglers: list storing Pauli words for construction of optimized qcc_circuit.
        params: parameters for the rotations
        barrier: barriers between blocks of gates
    Returns:
        qcc_circuit
    '''
    # assert the number of entanglers equals to the number of parameters
    assert(len(entanglers) == len(params))
    
    # check the format of the entanglers:
    entanglers = convert_pauli_list(entanglers)

    opt_qc = QuantumCircuit(len(entanglers[0]))
    append_clifford = QuantumCircuit(len(entanglers[0]))
    opt_paulis = entanglers.copy()
    opt_params = params.copy()

    #iterate over all the entanglers that needs optimization
    for opt_idx in range(len(entanglers)):
        
        #Extract the clifford for the current pauli with index
        extracted_clif = construct_Clifford_subcircuit(opt_paulis[opt_idx])
        #Add the extracted clifford to the beginning of the append_clifford at the end of the circuit
        append_clifford = extracted_clif.compose(append_clifford)
        #The extracted circuit for the current block with index
        extracted_qc = construct_extracted_subcircuit(entangler = opt_paulis[opt_idx], param = opt_params[opt_idx])
        #Add the extracted circuit to the optimized circuit
        if barrier == True:
            opt_qc.barrier()
        opt_qc.compose(extracted_qc, inplace = True)
        #If the current block is not the last block, iterate over the remaining blocks
        if opt_idx < len(entanglers) - 1:
            for pass_idx in range(opt_idx + 1, len(entanglers)):
                optimized_pauli_withsign = construct_opt_pauli(entangler = opt_paulis[pass_idx], clifford_circuit = extracted_clif)
                opt_sign = optimized_pauli_withsign[0:2]
                optimized_pauli = optimized_pauli_withsign[2:]
                #update the corresponding pauli and params in opt_paulis
                opt_paulis[pass_idx] = optimized_pauli
                if opt_sign == "-1":
                    opt_params[pass_idx] = -opt_params[pass_idx] 
    return opt_qc, append_clifford, opt_paulis, opt_params

def update_observables(Paulis: List[str], clifford_circuits: List[QuantumCircuit]):
    '''This function calcu;ates the pauli observable after extracting the clifford subcircuit.

    Args:
        Paulis: Pauli words for the observable
        clifford_circuit: the clifford subcircuit for extraction
    Returns:
        updated_Paulis
    '''
    num_qubits = len(Paulis[0])
    updated_Paulis = Paulis.copy()
    updated_signs = []
    for clifford in clifford_circuits:
        pauli_finder = ChecksFinder(num_qubits, clifford)
        for idx, pauli in enumerate(updated_Paulis):
            pauli_result = pauli_finder.find_checks_sym(pauli_group_elem = pauli)
            # print(updated_Paulis, pauli_result.p1_str, pauli_result.p2_str)
            if len(pauli_result.p1_str) > 3:
                updated_Paulis[idx] = pauli_result.p1_str[2:]
            else:
                 updated_Paulis[idx] = pauli_result.p1_str
            updated_signs.append(pauli_result.p1_str[0:2])
    return updated_signs, updated_Paulis

def simulate_expectationval(qc: QuantumCircuit, observable: str, shots: int=10000000):
    '''This function simulates the circuit and calculates the expectationvalue of the observable

    Args:
        qc: quantum circuit to be simulated
        observable: the pauli observable
    Returns:
        expectation_val: the expecation value
    '''
    pauli_commute = [[(observable, (1+0j))]]
    #generate individual measurment circuits that changes the basis
    meas_qcs = []
    for i in range(0, len(pauli_commute)):
        temp_qc = MeasureCircuit(pauli_commute[i], num_qubits = len(observable), num_qargs = len(observable))
        meas_qcs.append(temp_qc)
    qc_meas =  qc.compose(meas_qcs[0])
    qc_meas.measure_active()  
    simulator = AerSimulator()
    result = simulator.run(qc_meas, shots = shots).result()
    expectation_val = evaluation(result.get_counts(), shots = shots, Pauli = observable) 
    return expectation_val

    #this function generates the best CNOT tree structure given two pauli strings, maximize the cancellation
#TODO: these is a special case for the base entangler only contains one non identity string.
def find_single_tree(base_entangler_inv: str, match_entangler_inv: str):

    '''This function generates the best CNOT tree circuit for the base_entangler, maximizing the minimization of match_entangler.

    Args:
        base_entangler: the base entangler that searchers for the CNOT tree structure
        match_entangler: the target entangler that we are matching and minimizing
    Returns:
        CNOT_tree: the CNOT tree circuit
    '''
    num_qubits  = len(base_entangler_inv)
    I_list = []
    X_list = []
    Y_list = []
    Z_list = []
    N_list = []
    match_entangler = match_entangler_inv[::-1]
    base_entangler = base_entangler_inv[::-1]
    #first iterate over the base_entangler and find the non I index:
    for i in range(num_qubits):
        if base_entangler[i] != 'I':
            if match_entangler[i] == 'I':
                I_list.append(i)
            elif match_entangler[i] == 'X':
                X_list.append(i)
            elif match_entangler[i] == 'Y':
                Y_list.append(i)
            elif match_entangler[i] == 'Z':
                Z_list.append(i)
            else:
                raise Exception("Incorrect letter in entangler spotted", match_entangler)
        else:
            N_list.append(i)
    #Based on the lists, construct the quantum circuit:
    root_list = []
    qc = QuantumCircuit(num_qubits)
    I_root = X_root = Y_root = Z_root = N_root = final_root = -1
    #iterate over the I list:
    if len(I_list) == 1:
        I_root = I_list[0]
    elif len(I_list) > 1:
        for i_idx in range(len(I_list) - 1):
            qc.cx(I_list[i_idx], I_list[i_idx + 1])
        I_root = I_list[-1]
    #iterate over the Z list:
    if len(Z_list) == 1:
        Z_root = Z_list[0]
    elif len(Z_list) > 1:
        for z_idx in range(len(Z_list) - 1):
            qc.cx(Z_list[z_idx], Z_list[z_idx + 1])
        Z_root = Z_list[-1]
    #iterate over the X list:
    if len(X_list) == 1:
        X_root = X_list[0]
    elif len(X_list) > 1:
        for x_idx in range(len(X_list) - 1):
            qc.cx(X_list[x_idx], X_list[x_idx + 1])
        X_root = X_list[-1]
    #iterate over the Y list:
    if len(Y_list) == 1:
        Y_root = Y_list[0]
    elif len(Y_list) > 1:
        for y_idx in range(len(Y_list) - 1):
            qc.cx(Y_list[y_idx], Y_list[y_idx + 1])
        Y_root = Y_list[-1]
    # #iterate over the N list:
    # if len(N_list) == 1:
    #     N_root = N_list[0]
    # elif len(N_list) > 1:
    #     for n_idx in range(len(N_list) - 1):
    #         qc.cx(N_list[n_idx], N_list[n_idx + 1])
    #     N_root = N_list[-1]

    # # Finally, connect the I, Z, X roots with Y and N_root: XI, YZ, XY are preferred
    # if Z_root > -1:
    #     # If there is a Z root, find a Y root and connect; otherwise, connect with X, I root, or N__list
    #     if Y_root > -1:
    #         qc.cx(Z_root, Y_root)
    #     elif X_root > -1:
    #         qc.cx(Z_root, X_root)
    #     elif I_root > -1:
    #         qc.cx(Z_root, I_root)
    #     elif N_root > -1:
    #         qc.cx(Z_root, N_root)
    #     Z_root = -1

    #     if I_root == -1 and Y_root == -1 and X_root == -1 and N_root == -1:
    #         final_root = Z_root

    # if I_root > -1:
    #     #if there is I root, find X  and Y root and connect with I root:
    #     if X_root > -1:
    #         qc.cx(I_root, X_root)
    #     elif Y_root > -1:
    #         qc.cx(I_root, Y_root)
    #     elif N_root > -1:
    #         qc.cx(I_root, N_root)
    #     I_root = -1

    #     if X_root == -1 and Y_root == -1 and N_root == -1:
    #         final_root = I_root
    
    # if Y_root > -1:
    #     #if there is Y root, find I and X root and connect with Y root:
    #     if X_root > -1:
    #         qc.cx(Y_root, X_root)
    #     elif N_root > -1:
    #         qc.cx(Y_root, N_root)
    #     Y_root = -1  

    #     if X_root == -1 and N_root == -1:
    #         final_root = Y_root

    # if X_root > -1:
    #     if N_root > -1:
    #         qc.cx(X_root, N_root)
    #     X_root = -1

    #     if N_root == -1:
    #         final_root = X_root
    
    # if N_root > -1:
    #     final_root = N_root
    # Function to connect the roots based on priority and set the final root
    def connect_roots(qc, root_name, roots_dict, priorities):
        root_value = roots_dict[root_name]
        if root_value > -1:
            for other_root_name in priorities:
                if roots_dict[other_root_name] > -1:
                    qc.cx(root_value, roots_dict[other_root_name])
                    # roots_dict[other_root_name] = -1
                    roots_dict[root_name] = -1
                    return roots_dict[other_root_name]
        return -1

    # Initialize the root dictionary
    roots_dict = {
        "Z_root": Z_root,
        "I_root": I_root,
        "Y_root": Y_root,
        "X_root": X_root,
        "N_root": N_root
    }

    final_root = None

    # Connection priorities
    priority_connections = {
        "Z_root": ["Y_root", "X_root", "I_root"],
        "I_root": ["X_root", "Y_root"],
        "Y_root": ["X_root"],
        # "X_root": ["N_root"]
    }

    # Connect roots based on priority
    for root_name in ["Z_root", "I_root", "Y_root", "X_root"]:
        if root_name in priority_connections:
            last_connected = connect_roots(qc, root_name, roots_dict, priority_connections[root_name])
            if last_connected != -1:
                final_root = last_connected

    # If any roots are still unconnected, set final_root
    for root_name in ["Z_root", "I_root", "Y_root", "X_root"]:
        if roots_dict[root_name] > -1:
            final_root = roots_dict[root_name]
            # print("final, root, rootes_dict[root_name]", final_root, roots_dict[root_name])


    # Ensure final_root is set correctly if not already set
    if final_root is None:
        #final_root = X_root
        raise Exception("final root not set")


    return final_root, qc

#TODO: these is a special case for the base entangler only contains one non identity string.
def find_single_tree_lookahead(base_entangler_inv: str, match_entangler_inv: str, lookahead_entanglers_inv: List[str]):

    '''This function generates the best CNOT tree circuit for the base_entangler, maximizing the minimization of match_entangler.

    Args:
        base_entangler: the base entangler that searchers for the CNOT tree structure
        match_entangler: the target entangler that we are matching and minimizing
    Returns:
        CNOT_tree: the CNOT tree circuit
    '''
    num_qubits  = len(base_entangler_inv)
    I_list = []
    X_list = []
    Y_list = []
    Z_list = []
    N_list = []
    match_entangler = match_entangler_inv[::-1]
    base_entangler = base_entangler_inv[::-1]
    lookahead_entanglers = [entangler[::-1] for entangler in lookahead_entanglers_inv]

    # First, create a dictionary to store counts of each Pauli operator for each index
    pauli_counts = {i: {'I': 0, 'X': 0, 'Y': 0, 'Z': 0} for i in range(num_qubits)}

    # Iterate over lookahead_entanglers to fill in the pauli_counts
    for lookahead_pauli in lookahead_entanglers:
        for i in range(num_qubits):
            pauli_counts[i][lookahead_pauli[i]] += 1
    #print(lookahead_entanglers)
    #print("pauli_counts", pauli_counts)
    # Iterate over the base_entangler and find the non-I index
    for i in range(num_qubits):
        if base_entangler[i] != 'I':
            match_char = match_entangler[i]
            if match_char in pauli_counts[i]:
                count = pauli_counts[i][match_char]
                if match_char == 'I':
                    I_list.append([i, count])
                elif match_char == 'X':
                    X_list.append([i, count])
                elif match_char == 'Y':
                    Y_list.append([i, count])
                elif match_char == 'Z':
                    Z_list.append([i, count])
                else:
                    raise Exception("Incorrect letter in entangler spotted", match_entangler)
        else:
            N_list.append(i)
    # Sort the lists based on the count in descending order
    I_list.sort(key=lambda x: x[1], reverse=True)
    X_list.sort(key=lambda x: x[1], reverse=True)
    Y_list.sort(key=lambda x: x[1], reverse=True)
    Z_list.sort(key=lambda x: x[1], reverse=True)
    #print("base_entangler", I_list, X_list, Y_list, Z_list)

    #Based on the lists, construct the quantum circuit:
    root_list = []
    qc = QuantumCircuit(num_qubits)
    I_root = X_root = Y_root = Z_root = N_root = final_root = -1
    #iterate over the I list:
    if len(I_list) == 1:
        I_root = I_list[0][0]
    elif len(I_list) > 1:
        for i_idx in range(len(I_list) - 1):
            qc.cx(I_list[i_idx][0], I_list[i_idx + 1][0])
        I_root = I_list[-1][0]
    #iterate over the Z list:
    if len(Z_list) == 1:
        Z_root = Z_list[0][0]
    elif len(Z_list) > 1:
        for z_idx in range(len(Z_list) - 1):
            qc.cx(Z_list[z_idx][0], Z_list[z_idx + 1][0])
        Z_root = Z_list[-1][0]
    #iterate over the X list:
    if len(X_list) == 1:
        X_root = X_list[0][0]
    elif len(X_list) > 1:
        for x_idx in range(len(X_list) - 1):
            qc.cx(X_list[x_idx][0], X_list[x_idx + 1][0])
        X_root = X_list[-1][0]
    #iterate over the Y list:
    if len(Y_list) == 1:
        Y_root = Y_list[0][0]
    elif len(Y_list) > 1:
        for y_idx in range(len(Y_list) - 1):
            qc.cx(Y_list[y_idx][0], Y_list[y_idx + 1][0])
        Y_root = Y_list[-1][0]

    # Function to connect the roots based on priority and set the final root
    def connect_roots(qc, root_name, roots_dict, priorities):
        root_value = roots_dict[root_name]
        if root_value > -1:
            for other_root_name in priorities:
                if roots_dict[other_root_name] > -1:
                    qc.cx(root_value, roots_dict[other_root_name])
                    # roots_dict[other_root_name] = -1
                    roots_dict[root_name] = -1
                    return roots_dict[other_root_name]
        return -1

    # Initialize the root dictionary
    roots_dict = {
        "Z_root": Z_root,
        "I_root": I_root,
        "Y_root": Y_root,
        "X_root": X_root,
        "N_root": N_root
    }

    final_root = None

    # Connection priorities
    priority_connections = {
        "Z_root": ["Y_root", "X_root", "I_root"],
        "I_root": ["X_root", "Y_root"],
        "Y_root": ["X_root"],
        # "X_root": ["N_root"]
    }

    # Connect roots based on priority
    for root_name in ["Z_root", "I_root", "Y_root", "X_root"]:
        if root_name in priority_connections:
            last_connected = connect_roots(qc, root_name, roots_dict, priority_connections[root_name])
            if last_connected != -1:
                final_root = last_connected

    # If any roots are still unconnected, set final_root
    for root_name in ["Z_root", "I_root", "Y_root", "X_root"]:
        if roots_dict[root_name] > -1:
            final_root = roots_dict[root_name]
            # print("final, root, rootes_dict[root_name]", final_root, roots_dict[root_name])


    # Ensure final_root is set correctly if not already set
    if final_root is None:
        #final_root = X_root
        raise Exception("final root not set")


    return final_root, qc


        


def fc_tree_circuit(entanglers: List[str], params: List[float], barrier=False):
    '''This function defines the optimized fully connected tree block for hamtilbonian simulation. 
    
    Args:
        entanglers: list storing Pauli words for construction of optimized qcc_circuit.
        params: parameters for the rotations
        barrier: barriers between blocks of gates
    Returns:
        qcc_circuit
    '''
    # assert the number of entanglers equals to the number of parameters
    assert(len(entanglers) == len(params))
    
    # check the format of the entanglers:
    entanglers = convert_pauli_list(entanglers)

    opt_qc = QuantumCircuit(len(entanglers[0]))
    append_clifford = QuantumCircuit(len(entanglers[0]))
    opt_paulis = entanglers.copy()
    opt_params = params.copy()

    #iterate over all the entanglers that needs optimization
    for opt_idx in range(len(entanglers) - 1):
        #first push the single qubit gates through:
        sq_qc = construct_sq_subcircuit(opt_paulis[opt_idx])
        for pass_idx in range(opt_idx + 1, len(entanglers)):
            optimized_pauli_withsign = construct_opt_pauli(entangler = opt_paulis[pass_idx], clifford_circuit = sq_qc)
            opt_sign = optimized_pauli_withsign[0:2]
            optimized_pauli = optimized_pauli_withsign[2:]
            #update the corresponding pauli and params in opt_paulis
            opt_paulis[pass_idx] = optimized_pauli
            if opt_sign == "-1":
                opt_params[pass_idx] = -opt_params[pass_idx] 
        #Then construct the tree based on the next pauli: #TODO: this should be updated to based on all the next commuting Paulis
        # print("entanglers for tree", opt_paulis[opt_idx], opt_paulis[opt_idx+ 1])

        sq_index, init_cx_tree = find_single_tree(base_entangler_inv = opt_paulis[opt_idx], match_entangler_inv = opt_paulis[opt_idx + 1])
        init_clif = sq_qc.inverse()
        init_clif.compose(init_cx_tree, inplace = True)
        extracted_cx_tree = init_cx_tree.inverse()
        extracted_clif = init_clif.inverse()
        append_clifford = extracted_clif.compose(append_clifford)

        init_clif.rz(opt_params[opt_idx], sq_index)

        #Add the extracted circuit to the optimized circuit
        #print("after extract sq", opt_paulis)
        if barrier == True:
            opt_qc.barrier()
        opt_qc.compose(init_clif, inplace = True)
        #print(extracted_cx_tree)
        for pass_idx in range(opt_idx + 1, len(entanglers)):
            optimized_pauli_withsign = construct_opt_pauli(entangler = opt_paulis[pass_idx], clifford_circuit = extracted_cx_tree)
            opt_sign = optimized_pauli_withsign[0:2]
            optimized_pauli = optimized_pauli_withsign[2:]
            #update the corresponding pauli and params in opt_paulis
            opt_paulis[pass_idx] = optimized_pauli
            if opt_sign == "-1":
                opt_params[pass_idx] = -opt_params[pass_idx] 
        #print("after extract cx tree", opt_paulis)

    #extract for the last block:
    extracted_clif = construct_Clifford_subcircuit(opt_paulis[-1])
    #Add the extracted clifford to the beginning of the append_clifford at the end of the circuit
    append_clifford = extracted_clif.compose(append_clifford)
    #The extracted circuit for the current block with index
    extracted_qc = construct_extracted_subcircuit(entangler = opt_paulis[-1], param = opt_params[-1])
    #Add the extracted circuit to the optimized circuit
    # print("final paulis", opt_paulis)
    if barrier == True:
        opt_qc.barrier()
    opt_qc.compose(extracted_qc, inplace = True)
    return opt_qc, append_clifford, opt_paulis, opt_params


def generate_fc_tree_lookahead_circuit(entanglers: List[str], params: List[float], barrier=False, lookahead_size=10):
    '''This function defines the optimized fully connected tree block for hamtilbonian simulation. 
    
    Args:
        entanglers: list storing Pauli words for construction of optimized qcc_circuit.
        params: parameters for the rotations
        barrier: barriers between blocks of gates
    Returns:
        qcc_circuit
    '''
    # assert the number of entanglers equals to the number of parameters
    assert(len(entanglers) == len(params))
    
    # check the format of the entanglers:
    entanglers = convert_pauli_list(entanglers)

    opt_qc = QuantumCircuit(len(entanglers[0]))
    append_clifford = QuantumCircuit(len(entanglers[0]))
    opt_paulis = entanglers.copy()
    opt_params = params.copy()

    #iterate over all the entanglers that needs optimization
    for opt_idx in range(len(entanglers) - 1):
        #first push the single qubit gates through:
        sq_qc = construct_sq_subcircuit(opt_paulis[opt_idx])
        for pass_idx in range(opt_idx + 1, len(entanglers)):
            optimized_pauli_withsign = construct_opt_pauli(entangler = opt_paulis[pass_idx], clifford_circuit = sq_qc)
            opt_sign = optimized_pauli_withsign[0:2]
            optimized_pauli = optimized_pauli_withsign[2:]
            #update the corresponding pauli and params in opt_paulis
            opt_paulis[pass_idx] = optimized_pauli
            if opt_sign == "-1":
                opt_params[pass_idx] = -opt_params[pass_idx] 
        #Then construct the tree based on the next pauli: #TODO: this should be updated to based on all the next commuting Paulis
        # print("entanglers for tree", opt_paulis[opt_idx], opt_paulis[opt_idx+ 1])
        end_idx = min(opt_idx + 2 + lookahead_size, len(opt_paulis))
        lookahead_entanglers = opt_paulis[opt_idx + 2: end_idx]
        sq_index, init_cx_tree = find_single_tree_lookahead(base_entangler_inv = opt_paulis[opt_idx], match_entangler_inv = opt_paulis[opt_idx + 1], lookahead_entanglers_inv=lookahead_entanglers)
        init_clif = sq_qc.inverse()
        init_clif.compose(init_cx_tree, inplace = True)
        extracted_cx_tree = init_cx_tree.inverse()
        extracted_clif = init_clif.inverse()
        append_clifford = extracted_clif.compose(append_clifford)

        init_clif.rz(opt_params[opt_idx], sq_index)

        #Add the extracted circuit to the optimized circuit
        # print("after extract sq", opt_paulis)
        if barrier == True:
            opt_qc.barrier()
        opt_qc.compose(init_clif, inplace = True)
        # print(extracted_cx_tree)
        for pass_idx in range(opt_idx + 1, len(entanglers)):
            optimized_pauli_withsign = construct_opt_pauli(entangler = opt_paulis[pass_idx], clifford_circuit = extracted_cx_tree)
            opt_sign = optimized_pauli_withsign[0:2]
            optimized_pauli = optimized_pauli_withsign[2:]
            #update the corresponding pauli and params in opt_paulis
            opt_paulis[pass_idx] = optimized_pauli
            if opt_sign == "-1":
                opt_params[pass_idx] = -opt_params[pass_idx] 
        # print("after extract cx tree", opt_paulis)

    #extract for the last block:
    extracted_clif = construct_Clifford_subcircuit(opt_paulis[-1])
    #Add the extracted clifford to the beginning of the append_clifford at the end of the circuit
    append_clifford = extracted_clif.compose(append_clifford)
    #The extracted circuit for the current block with index
    extracted_qc = construct_extracted_subcircuit(entangler = opt_paulis[-1], param = opt_params[-1])
    #Add the extracted circuit to the optimized circuit
    # print("final paulis", opt_paulis)
    if barrier == True:
        opt_qc.barrier()
    opt_qc.compose(extracted_qc, inplace = True)
    return opt_qc, append_clifford, opt_paulis, opt_params


def convert_commute_paulis(Paulis: List[str]) -> List[List[str]]:
    '''This function converts the Paulis to commute sets.
    
    Args:
        Paulis: list storing Pauli words for construction of optimized circuit.
    Returns:
        paulis_sets: list of Paul sets that commutes
    '''
    current_set = []
    paulis_sets = []
    
    for idx in range(len(Paulis)):
        pauli = Paulis[idx]
        if not current_set:
            current_set.append(pauli)
        else:
            can_be_added = True
            for current_pauli in current_set:
                if not pauli_strings_commute(current_pauli, pauli):
                    can_be_added = False
                    break
            if can_be_added:
                current_set.append(pauli)
            else:
                paulis_sets.append(current_set)
                current_set = [pauli]

    if current_set:
        paulis_sets.append(current_set)

    return paulis_sets


def convert_commute_sets(Paulis: List[str], params: List[float]) -> List[List[str]]:
    '''This function converts the Paulis to commute sets.
    
    Args:
        Paulis: list storing Pauli words for construction of optimized circuit.
    Returns:
        paulis_sets: list of Paul sets that commutes
    '''
    current_set = []
    paulis_sets = []
    
    for idx in range(len(Paulis)):
        pauli = Paulis[idx]
        if not current_set:
            current_set.append([pauli, params[idx]])
        else:
            can_be_added = True
            for current_pauli in current_set:
                if not pauli_strings_commute(current_pauli[0], pauli):
                    can_be_added = False
                    break
            if can_be_added:
                current_set.append([pauli, params[idx]])
            else:
                paulis_sets.append(current_set)
                current_set = [[pauli, params[idx]]]

    if current_set:
        paulis_sets.append(current_set)

    return paulis_sets


def pauli_weight(pauli: str) -> int:
    weight = 0
    valid_paulis = {'I', 'X', 'Y', 'Z'}
    
    for char in pauli:
        if char not in valid_paulis:
            raise ValueError(f"Invalid character '{char}' in Pauli string: {pauli}")
        if char != 'I':
            weight += 1
    
    return weight

def calculate_opt_weight(base_entangler_inv: str, match_entangler_inv: str) -> int:
    '''This function generates the best CNOT tree circuit for the base_entangler and calculates the weight of match-entangler after optimization
    , maximizing the minimization of match_entangler.
    Args:
        base_entangler: the base entangler that searchers for the CNOT tree structure
        match_entangler: the target entangler that we are matching and minimizing
    Returns:
        opt_weight: the weight of the match_entangler_inv after optimization
    '''
    #TODO: this should be optimized to simply based on the two strings instead of based on circuits.
    #assume we have already pushed the single qubit gates
    sq_index, init_cx_tree = find_single_tree(base_entangler_inv, match_entangler_inv)
    extracted_cx_tree = init_cx_tree.inverse()
    optimized_pauli_withsign = construct_opt_pauli(entangler = match_entangler_inv, clifford_circuit = extracted_cx_tree)
    optimized_pauli = optimized_pauli_withsign[2:]
    return pauli_weight(optimized_pauli), optimized_pauli_withsign


def find_best_pauli(base_entangler: str, commute_sets: List[List[str]]) -> int:
    '''This function finds the best pauli entangler in a set of commuting paulis.
    Args:
        base_entangler: the base entangler that searchers for the CNOT tree structure
        commute_entanglers: the target entangler that we are matching and minimizing
    Returns:
        ordered_entanglers: the ordered commute_entanglers after optimization
    '''
    ordered_entanglers = commute_sets.copy()
    min_weight = float('inf')
    min_index = None
    for idx, entangler_set in enumerate(ordered_entanglers):
        weight, optimized_pauli_withsign = calculate_opt_weight(base_entangler, entangler_set[0])
        logger.debug('weight:%s, entangler_set:%s, optimized_pauli_withsign:%s', weight, entangler_set, optimized_pauli_withsign)
        if weight < min_weight:
            min_weight = weight
            min_index = idx
    # Remove the element at the specified index
    element = ordered_entanglers.pop(min_index)
    # Insert the element at the beginning of the list
    ordered_entanglers.insert(0, element)

    return ordered_entanglers



def estimate_fc_reduction(entangler: str, opt_entanglers: List[str]):
    #TODO: have not finished the calculation
    weight_indexes = []
    valid_paulis = {'I', 'X', 'Y', 'Z'}
    
    for idx, char in enumerate(entangler):
        if char not in valid_paulis:
            raise ValueError(f"Invalid character '{char}' in Pauli string: {pauli}")
        if char != 'I':
            weight_indexes.append(idx)
    
    for pauli in opt_entanglers:
        z_count = x_count = y_count = i_count = 0
        for index in weight_indexes:
            if entangler[index] == 'X':
                if pauli[index] == 'X':
                    z_count += 1
                elif pauli[index] == 'Y':
                    y_count += 1
                elif pauli[index] == 'Z':
                    x_count += 1
                elif pauli[index] == 'I':
                    i_count += 1
            elif entangler[index] == 'Y':
                if pauli[index] == 'X':
                    y_count += 1
                elif pauli[index] == 'Y':
                    z_count += 1
                elif pauli[index] == 'Z':
                    x_count += 1
                elif pauli[index] == 'I':
                    i_count += 1
            elif entangler[index] == 'Z':
                if pauli[index] == 'X':
                    x_count += 1
                elif pauli[index] == 'Y':
                    y_count += 1
                elif pauli[index] == 'Z':
                    z_count += 1
                elif pauli[index] == 'I':
                    i_count += 1
        cost = 0
        if y_count > 0:
            z_count = 0
        else:
            z_count = min(z_count, 1)
        if x_count > 0:
            x_count = x_count // 2
    return None


def fc_tree_commute_circuit(entanglers: List[str], params: List[float], barrier=False):
    '''This function defines the optimized fully connected tree block for hamtilbonian simulation in commute list format
    
    Args:
        entanglers: list storing Pauli words for construction of optimized qcc_circuit.
        params: parameters for the rotations
        barrier: barriers between blocks of gates
    Returns:
        opt_qc, append_clifford, opt_paulis, opt_params
    '''
    commute_sets = convert_commute_sets(Paulis= entanglers, params = params)

    opt_qc = QuantumCircuit(len(entanglers[0]))
    append_clifford = QuantumCircuit(len(entanglers[0]))
    # opt_params = params.copy()
    sorted_entanglers = []

    #sort all the paulis based on their weight:#TODO: need to resolve when two cases have the same weight
    for commute_list in commute_sets:
        sorted_list = sorted(commute_list, key=lambda x: pauli_weight(x[0]))
        sorted_entanglers.append(sorted_list)

    logging.debug("start_sorted_list: %s", sorted_entanglers)
    # Iterate over all the lists of commuting entanglers that need optimization
    for commute_idx, sorted_list in enumerate(sorted_entanglers):
        # updated_commute_list = [sorted_list[0]]  # Initialize with the first Pauli string
        # remaining_commute_paulis = sorted_list[1:].copy()
        
        for pauli_idx in range(len(sorted_list)):
            #here we start process for each current pauli
            curr_pauli = sorted_entanglers[commute_idx][pauli_idx][0]
            curr_param = sorted_entanglers[commute_idx][pauli_idx][1]
            sq_qc = construct_sq_subcircuit(curr_pauli)  # Construct the single qubit subcircuit
            #update all the following paulis with single qubit gates:
            for set_idx in range(commute_idx, len(sorted_entanglers)):
                start_idx = pauli_idx + 1 if set_idx == commute_idx else 0
                #the following code automaticaly skip the iteration when this pauli is at the end of the current group
                for k in range(start_idx, len(sorted_entanglers[set_idx])):
                    optimized_pauli_withsign = construct_opt_pauli(
                        entangler=sorted_entanglers[set_idx][k][0], 
                        clifford_circuit=sq_qc
                    )
                    opt_sign = optimized_pauli_withsign[:2]
                    optimized_pauli = optimized_pauli_withsign[2:]
                    
                    # Update the corresponding Pauli and parameters in opt_paulis
                    sorted_entanglers[set_idx][k][0] = optimized_pauli
                    if opt_sign == "-1":
                        sorted_entanglers[set_idx][k][1] = -sorted_entanglers[set_idx][k][1]
            logging.debug("after_extract_sq: %s", sorted_entanglers)
            #find the best next pauli:
            #based on the pauli_idx determine which should be the next pauli, or the pauli in the next commuting list
            if pauli_idx == len(sorted_list) - 1: # if this pauli is the last one in a commuting list, find the pauli in the next list
                if commute_idx == len(sorted_entanglers) - 1:
                    next_pauli = None
                else:
                    logging.debug("pauli_idx: %s, sorted_list: %s", pauli_idx, sorted_list)
                    ordered_entanglers = find_best_pauli(base_entangler = curr_pauli, commute_sets = sorted_entanglers[commute_idx + 1])
                    next_pauli = ordered_entanglers[0][0]
                    sorted_entanglers[commute_idx + 1] = ordered_entanglers
                    logging.debug("ordered_entanglers in next: %s",ordered_entanglers)

            else: # search within the current commuting list:
                # if len(remaining_commute_paulis) == 0:
                #     next_pauli = None
                # else:
                ordered_entanglers = find_best_pauli(base_entangler = curr_pauli, commute_sets = sorted_entanglers[commute_idx][pauli_idx + 1:])
                next_pauli = ordered_entanglers[0][0]
                sorted_entanglers[commute_idx][pauli_idx + 1:] = ordered_entanglers
            logging.debug("after_search_for_the best next: %s", sorted_entanglers)
            logging.debug("next_pauli: %s", next_pauli)
            
            if next_pauli == None:      
                #extract for the last block:
                extracted_clif = construct_Clifford_subcircuit(curr_pauli)
                #Add the extracted clifford to the beginning of the append_clifford at the end of the circuit
                append_clifford = extracted_clif.compose(append_clifford)
                #The extracted circuit for the current block with index
                extracted_qc = construct_extracted_subcircuit(entangler = curr_pauli, param = curr_param)
                #Add the extracted circuit to the optimized circuit
                logging.debug("final paulis: %s", curr_pauli)

                if barrier == True:
                    opt_qc.barrier()
                opt_qc.compose(extracted_qc, inplace = True)

            else:
                sq_index, init_cx_tree = find_single_tree(base_entangler_inv = curr_pauli, match_entangler_inv = next_pauli)
                init_clif = sq_qc.inverse()
                init_clif.compose(init_cx_tree, inplace = True)
                extracted_cx_tree = init_cx_tree.inverse()
                extracted_clif = init_clif.inverse()
                #Add the extracted_clifford to the append clifford
                append_clifford = extracted_clif.compose(append_clifford)

                init_clif.rz(curr_param, sq_index)

                #Add the extracted circuit to the optimized circuit
                if barrier == True:
                    opt_qc.barrier()
                opt_qc.compose(init_clif, inplace = True)


                #update all the following paulis with cx tree gates:
                for set_idx in range(commute_idx, len(sorted_entanglers)):
                    start_idx = pauli_idx + 1 if set_idx == commute_idx else 0
                    #the following code automaticaly skip the iteration when this pauli is at the end of the current group
                    for k in range(start_idx, len(sorted_entanglers[set_idx])):
                        optimized_pauli_withsign = construct_opt_pauli(entangler=sorted_entanglers[set_idx][k][0] , clifford_circuit = extracted_cx_tree)
                        opt_sign = optimized_pauli_withsign[0:2]
                        optimized_pauli = optimized_pauli_withsign[2:]
                        #update the corresponding pauli and params in opt_paulis
                        sorted_entanglers[set_idx][k][0] = optimized_pauli
                        if opt_sign == "-1":
                            sorted_entanglers[set_idx][k][1] = -sorted_entanglers[set_idx][k][1]
                logging.debug("after extract cx tree: %s", sorted_entanglers)




    return opt_qc, append_clifford, sorted_entanglers


def find_next_k_paulis(sorted_entanglers, commute_idx, pauli_idx, K):
    result = []
    count = 0

    for set_idx in range(commute_idx, len(sorted_entanglers)):
        start_idx = pauli_idx + 1 if set_idx == commute_idx else 0

        # Iterate through the current group to find the next K Pauli strings
        for k in range(start_idx, len(sorted_entanglers[set_idx])):
            result.append(sorted_entanglers[set_idx][k][0])
            count += 1

            # Break if we have found K Pauli strings
            if count == K:
                return result

    return result

def fc_tree_commute_lookahead_circuit(entanglers: List[str], params: List[float], barrier=False, lookahead_size=10):
    '''This function defines the optimized fully connected tree block for hamtilbonian simulation in commute list format, also considering lookahead
    
    Args:
        entanglers: list storing Pauli words for construction of optimized qcc_circuit.
        params: parameters for the rotations
        barrier: barriers between blocks of gates
    Returns:
        opt_qc, append_clifford, opt_paulis, opt_params
    '''
    commute_sets = convert_commute_sets(Paulis= entanglers, params = params)

    opt_qc = QuantumCircuit(len(entanglers[0]))
    append_clifford = QuantumCircuit(len(entanglers[0]))
    # opt_params = params.copy()
    sorted_entanglers = []

    #sort all the paulis based on their weight:#TODO: need to resolve when two cases have the same weight
    for commute_list in commute_sets:
        sorted_list = sorted(commute_list, key=lambda x: pauli_weight(x[0]))
        sorted_entanglers.append(sorted_list)

    logging.debug("start_sorted_list: %s", sorted_entanglers)
    # Iterate over all the lists of commuting entanglers that need optimization
    for commute_idx, sorted_list in enumerate(sorted_entanglers):
        # updated_commute_list = [sorted_list[0]]  # Initialize with the first Pauli string
        # remaining_commute_paulis = sorted_list[1:].copy()
        
        for pauli_idx in range(len(sorted_list)):
            #here we start process for each current pauli
            curr_pauli = sorted_entanglers[commute_idx][pauli_idx][0]
            curr_param = sorted_entanglers[commute_idx][pauli_idx][1]
            sq_qc = construct_sq_subcircuit(curr_pauli)  # Construct the single qubit subcircuit
            #update all the following paulis with single qubit gates:
            for set_idx in range(commute_idx, len(sorted_entanglers)):
                start_idx = pauli_idx + 1 if set_idx == commute_idx else 0
                #the following code automaticaly skip the iteration when this pauli is at the end of the current group
                for k in range(start_idx, len(sorted_entanglers[set_idx])):
                    optimized_pauli_withsign = construct_opt_pauli(
                        entangler=sorted_entanglers[set_idx][k][0], 
                        clifford_circuit=sq_qc
                    )
                    opt_sign = optimized_pauli_withsign[:2]
                    optimized_pauli = optimized_pauli_withsign[2:]
                    
                    # Update the corresponding Pauli and parameters in opt_paulis
                    sorted_entanglers[set_idx][k][0] = optimized_pauli
                    if opt_sign == "-1":
                        sorted_entanglers[set_idx][k][1] = -sorted_entanglers[set_idx][k][1]
            logging.debug("after_extract_sq: %s", sorted_entanglers)
            #find the best next pauli:
            #based on the pauli_idx determine which should be the next pauli, or the pauli in the next commuting list
            if pauli_idx == len(sorted_list) - 1: # if this pauli is the last one in a commuting list, find the pauli in the next list
                if commute_idx == len(sorted_entanglers) - 1:
                    next_pauli = None
                else:
                    logging.debug("pauli_idx: %s, sorted_list: %s", pauli_idx, sorted_list)
                    ordered_entanglers = find_best_pauli(base_entangler = curr_pauli, commute_sets = sorted_entanglers[commute_idx + 1])
                    next_pauli = ordered_entanglers[0][0]
                    sorted_entanglers[commute_idx + 1] = ordered_entanglers
                    logging.debug("ordered_entanglers in next: %s",ordered_entanglers)

            else: # search within the current commuting list:
                # if len(remaining_commute_paulis) == 0:
                #     next_pauli = None
                # else:
                ordered_entanglers = find_best_pauli(base_entangler = curr_pauli, commute_sets = sorted_entanglers[commute_idx][pauli_idx + 1:])
                next_pauli = ordered_entanglers[0][0]
                sorted_entanglers[commute_idx][pauli_idx + 1:] = ordered_entanglers
            logging.debug("after_search_for_the best next: %s", sorted_entanglers)
            logging.debug("next_pauli: %s", next_pauli)
            
            if next_pauli == None:      
                #extract for the last block:
                extracted_clif = construct_Clifford_subcircuit(curr_pauli)
                #Add the extracted clifford to the beginning of the append_clifford at the end of the circuit
                append_clifford = extracted_clif.compose(append_clifford)
                #The extracted circuit for the current block with index
                extracted_qc = construct_extracted_subcircuit(entangler = curr_pauli, param = curr_param)
                #Add the extracted circuit to the optimized circuit
                logging.debug("final paulis: %s", curr_pauli)

                if barrier == True:
                    opt_qc.barrier()
                opt_qc.compose(extracted_qc, inplace = True)

            else:
                lookahead_entanglers = find_next_k_paulis(sorted_entanglers, commute_idx, pauli_idx, lookahead_size)
                sq_index, init_cx_tree = find_single_tree_lookahead(base_entangler_inv = curr_pauli, match_entangler_inv = next_pauli, lookahead_entanglers_inv = lookahead_entanglers)
                init_clif = sq_qc.inverse()
                init_clif.compose(init_cx_tree, inplace = True)
                extracted_cx_tree = init_cx_tree.inverse()
                extracted_clif = init_clif.inverse()
                #Add the extracted_clifford to the append clifford
                append_clifford = extracted_clif.compose(append_clifford)

                init_clif.rz(curr_param, sq_index)

                #Add the extracted circuit to the optimized circuit
                if barrier == True:
                    opt_qc.barrier()
                opt_qc.compose(init_clif, inplace = True)


                #update all the following paulis with cx tree gates:
                for set_idx in range(commute_idx, len(sorted_entanglers)):
                    start_idx = pauli_idx + 1 if set_idx == commute_idx else 0
                    #the following code automaticaly skip the iteration when this pauli is at the end of the current group
                    for k in range(start_idx, len(sorted_entanglers[set_idx])):
                        optimized_pauli_withsign = construct_opt_pauli(entangler=sorted_entanglers[set_idx][k][0] , clifford_circuit = extracted_cx_tree)
                        opt_sign = optimized_pauli_withsign[0:2]
                        optimized_pauli = optimized_pauli_withsign[2:]
                        #update the corresponding pauli and params in opt_paulis
                        sorted_entanglers[set_idx][k][0] = optimized_pauli
                        if opt_sign == "-1":
                            sorted_entanglers[set_idx][k][1] = -sorted_entanglers[set_idx][k][1]
                logging.debug("after extract cx tree: %s", sorted_entanglers)




    return opt_qc, append_clifford, sorted_entanglers

            
    



        
    




