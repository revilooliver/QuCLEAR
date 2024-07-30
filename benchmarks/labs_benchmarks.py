import json
from qokit.labs import get_gate_optimized_terms_greedy


def paulis_mixer(num_qubits):
    '''Returns the list of Pauli strings for the mixer Hamiltonian.'''
    hamiltonian=[]
    for idx in range(num_qubits):
        pauli_str=list("I"*num_qubits)
        pauli_str[idx]="X"
        hamiltonian.append("".join(pauli_str))
    return hamiltonian

def labs_pauli_layers(num_qubits, layers, seed):
    '''Returns the list of cost Hamiltonian followed by the mixer Hamiltonian repeated
    for the desired layers.
    Args:
        num_qubits: number of qubits in circuit.
        layers: number of QAOA layers.
        seed: seed for numpy random number generator'''
    terms = get_gate_optimized_terms_greedy(num_qubits, seed=seed) # indices of Pauli Zs
    print(f"terms {terms}")
    cost_hamiltonian=[]
    for t in terms:
        pauli_str=list("I"*num_qubits)
        for idx in t:
            pauli_str[idx]="Z"
        cost_hamiltonian.append("".join(pauli_str))
    return [cost_hamiltonian, paulis_mixer(num_qubits)]*layers
    

def save_to_json(data, file=None):
    if file==None:
        file="labs_paulis/labs_paulis.json"
    else:
        file="labs_paulis/"+file

    with open(file, "w") as f:
        json.dump(data, f)