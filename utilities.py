import re
import random
from itertools import product
from typing import List
import os

def load_sycamore_coupling_map():
    reduced = True
    pth = os.path.join('sycamore_64.txt')

    coupling = []
    n = 0
    with open(pth, 'r') as file:
        lines = file.readlines()
        num_nodes, num_edges = map(int, lines[0].split()[:2])
        n = num_nodes
        
        # Add edges to the graph
        for edge in lines[1:]:
            node1, node2 = map(int, edge.split()[:2])
            coupling.append([node1, node2])
            coupling.append([node2, node1])

    return coupling

def generate_meas_strings(n):
    # Define the possible Pauli operators excluding the all-identity string
    paulis = ['Z']
    
    # Generate all possible n-qubit Pauli strings excluding the all-identity string
    pauli_strings = []
    for p in product(['I'] + paulis, repeat=n):
        string = ''.join(p)
        if 'X' in string or 'Y' in string or 'Z' in string:  # Exclude all-identity strings
            pauli_strings.append(string)
    
    return pauli_strings

#generate all possible pauli pairs except for all I:
def generate_pauli_strings(n):
    # Define the possible Pauli operators excluding the all-identity string
    paulis = ['X', 'Y', 'Z']
    
    # Generate all possible n-qubit Pauli strings excluding the all-identity string
    pauli_strings = []
    for p in product(['I'] + paulis, repeat=n):
        string = ''.join(p)
        if 'X' in string or 'Y' in string or 'Z' in string:  # Exclude all-identity strings
            pauli_strings.append(string)
    
    return pauli_strings

def generate_pauli_pairs(n):
    pauli_strings = generate_pauli_strings(n)
    
    # Generate pairs of Pauli strings
    pauli_pairs = [[p1, p2] for p1 in pauli_strings for p2 in pauli_strings]
    
    return pauli_pairs

def convert_pauli_list(entanglers: List[str]):
    '''This function checkes if the entanglers are in correct format without the signs.
    
    Args:
        entanglers: list storing Pauli words for construction of optimized qcc_circuit.
    Returns:
        paulis_list: list of Paulis with sign
    '''
    paulis_list = []
    for pauli in entanglers:
        if re.match(r"^[IXYZ]*$", pauli):
            paulis_list.append(pauli)
        # elif re.match(r"^[+-]1[IXYZ]*$", pauli):
        #     paulis_list.append(pauli)
        else:
            raise Exception("Incorrect Pauli spotted", pauli)
    return paulis_list

def pauli_strings_commute(pauli_str1, pauli_str2):
    """
    Determine if two Pauli strings commute.
    
    :param pauli_str1: A string representing the first Pauli operator.
    :param pauli_str2: A string representing the second Pauli operator.
    :return: True if the Pauli strings commute, False otherwise.
    """
    if len(pauli_str1) != len(pauli_str2):
        raise ValueError("Pauli strings must be of the same length.", pauli_str1, pauli_str2)
    
    commute = True  # Assume they commute until proven otherwise
    
    anticommute_count = 0
    
    for i in range(len(pauli_str1)):
        if pauli_str1[i] != pauli_str2[i] and pauli_str1[i] != 'I' and pauli_str2[i] != 'I':
            # Found anti-commuting Pauli matrices
            commute = False
            anticommute_count += 1

    if anticommute_count % 2 == 0:
        commute = True
    
    return commute


def generate_random_pauli_string(num_qubits):
    # Define the possible Pauli operators
    paulis = ['I', 'X', 'Y', 'Z']
    
    # Generate a random Pauli string
    return ''.join(random.choice(paulis) for _ in range(num_qubits))

def generate_pauli_strings(num_qubits, num_pauli_strings):
    # Generate the specified number of Pauli strings
    return [generate_random_pauli_string(num_qubits) for _ in range(num_pauli_strings)]

def compare_lists(list1, list2, tolerance=0.01):
    if len(list1) != len(list2):
        raise ValueError("The lists are of different lengths.")
    
    for index, (a, b) in enumerate(zip(list1, list2)):
        if abs(a - b) > tolerance:
            raise ValueError(f"Difference between elements at index {index} is greater than {tolerance}: {a} vs {b}")