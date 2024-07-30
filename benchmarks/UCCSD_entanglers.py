import numpy as np
import pennylane as qml
from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner
import json
def generate_UCCSD_entanglers(electrons, orbitals):
    # electrons_list = [2, 2, 4, 6, 8, 10, 12]
    # orbitals_list = [4, 6, 8, 12, 16, 20, 24]
    singles, doubles = qml.qchem.excitations(electrons, orbitals)
    UCCSD_Paulis = []

    # first we compute all the Paulis from fermionic single excitation
    for i in singles:
        Paulis = jordan_wigner(FermionOperator(str(i[1])+'^ ' + str(i[0])))
        
        # The Paulis.terms save the Jordan-Wigner transformed single excitations in dictionary format:
        # key (tuple of tuples): A dictionary storing the coefficients of the terms in the operator. 
        # The keys are the terms. A term is a product of individual factors; each factor is represented
        # by a tuple of the form (index, action), and these tuples are collected into a larger tuple
        # which represents the term as the product of its factors.
        UCCSD_Paulis.append(Paulis.terms)
        
    for i in doubles:
        Paulis = jordan_wigner(FermionOperator(str(i[2])+'^ '+str(i[3])+'^ '+str(i[1])+' '+str(i[0])))
        UCCSD_Paulis.append(Paulis.terms)
    entanglers = [] # list of lists

    # rewrite the dictionary key into entangler form:
    for Paulis_dict in UCCSD_Paulis:
        entangler_excitation = []
        
        for Paulis in list(Paulis_dict.keys()):    
            entangler = 'I' * orbitals
            entangler_list = list(entangler)
            for Pauli_tuple in Paulis:
                entangler_list[Pauli_tuple[0]] = Pauli_tuple[1]
            entangler = ''.join(entangler_list)
            entangler_excitation.append(entangler)
        
        entanglers.append(entangler_excitation)
    

    flattened_entanglers = [item for sublist in entanglers for item in sublist]
    return flattened_entanglers


def generate_UCCSD_entanglers_blocks():
    '''For tetris. We need the Hamiltonians in blocks.'''
    electrons_list = [2, 2, 4, 6, 8, 10, 12]
    orbitals_list = [4, 6, 8, 12, 16, 20, 24]
    for electrons, orbitals in zip(electrons_list, orbitals_list):
        singles, doubles = qml.qchem.excitations(electrons, orbitals)
        UCCSD_Paulis = []

        # first we compute all the Paulis from fermionic single excitation
        for i in singles:
            Paulis = jordan_wigner(FermionOperator(str(i[1])+'^ ' + str(i[0])))
            
            # The Paulis.terms save the Jordan-Wigner transformed single excitations in dictionary format:
            # key (tuple of tuples): A dictionary storing the coefficients of the terms in the operator. 
            # The keys are the terms. A term is a product of individual factors; each factor is represented
            # by a tuple of the form (index, action), and these tuples are collected into a larger tuple
            # which represents the term as the product of its factors.
            UCCSD_Paulis.append(Paulis.terms)
            
        for i in doubles:
            Paulis = jordan_wigner(FermionOperator(str(i[2])+'^ '+str(i[3])+'^ '+str(i[1])+' '+str(i[0])))
            UCCSD_Paulis.append(Paulis.terms)
        entanglers = [] # list of lists

        # rewrite the dictionary key into entangler form:
        for Paulis_dict in UCCSD_Paulis:
            entangler_excitation = []
            
            for Paulis in list(Paulis_dict.keys()):    
                entangler = 'I' * orbitals
                entangler_list = list(entangler)
                for Pauli_tuple in Paulis:
                    entangler_list[Pauli_tuple[0]] = Pauli_tuple[1]
                entangler = ''.join(entangler_list)
                entangler_excitation.append(entangler)
            
            entanglers.append(entangler_excitation)
        file_name=f"uccsd_hamiltonian_e{electrons}_o{orbitals}.json"
        with open(f'uccsd_hamiltonians/' + file_name, 'w') as paulis_file:
            json.dump(entanglers, paulis_file, indent=4)