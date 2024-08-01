from copy import deepcopy
import rustworkx as rx
from qiskit.quantum_info import SparsePauliOp
from rustworkx.visualization import mpl_draw as draw_graph
import json
import networkx
from rustworkx import networkx_converter, is_connected
import random as rnd

def paulis_mixer(nqubits):
    '''Returns the list of Pauli strings for the mixer Hamiltonian.'''
    hamiltonian=[]
    all_identity=list("I"*nqubits)
    for idx in range(nqubits):
        pauli_str=deepcopy(all_identity)
        pauli_str[idx]="X"
        hamiltonian.append("".join(pauli_str))
    return hamiltonian

def max_cut_pauli_layers(graph: rx.PyGraph, totter_number):
    '''Returns a list of a cost hamiltonian followed by a mixer with the specified number of layers.
    Args: graph: the graph for max cut.
        totter_number: the number of qaoa layers.'''
    max_cut_paulis = build_max_cut_paulis(graph)
    print(max_cut_paulis)
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
    cost_hamiltonian_pauli=cost_hamiltonian.paulis.to_labels() 
    hamiltonian_mixer=paulis_mixer(cost_hamiltonian.num_qubits)
    # return list(itertools.chain(*all_hams)) #if you want a flat list.
    return [cost_hamiltonian_pauli, hamiltonian_mixer]*totter_number

def build_graph(n, number_edges) -> rx.PyGraph:
    '''Creates a random graph of n nodes with number_edges edges. Undirected, not a multigraph,
    and no self loops. Outputs an Erdős-Rényi graph.'''
    graph=rx.undirected_gnm_random_graph(n, number_edges, seed=0) #fixed seed with given edge is reporducable
    for u, v in graph.edge_list():
        graph.update_edge(u,v, 1.0)
    draw_graph(graph,node_size=600, with_labels=True)
    return graph

def build_random_connected_graph(n) -> rx.PyGraph:
    '''Creates a random graph of n nodes with number_edges edges. Connected, undirected, not a multigraph,
    and no self loops.'''
    while True:
        number_edges=rnd.randint(n-1, int(n*(n-1)/2))
        graph=rx.undirected_gnm_random_graph(n, number_edges, seed=0) #fixed seed with given edge is reporducable
        if is_connected(graph):
            break
    for u, v in graph.edge_list():
        graph.update_edge(u,v, 1.0)
    draw_graph(graph,node_size=600, with_labels=True)
    return graph, number_edges

def build_regular_graph(nodes, edges) -> rx.PyGraph:
    '''Creates a random regular graph of n nodes with number_edges edges. Undirected, not a multigraph,
    and no self loops.'''
    graph=networkx.random_regular_graph(edges, nodes)
    graph=networkx_converter(graph)
    for u, v in graph.edge_list():
        graph.update_edge(u,v, 1.0)
    draw_graph(graph,node_size=600, with_labels=True)
    return graph

def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    """Convert the graph to Pauli list. Creates the cost Hamiltonian.
    """
    pauli_list = []
    for edge in list(graph.edge_list()):
        paulis = ["I"] * len(graph)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

        weight = graph.get_edge_data(edge[0], edge[1])

        pauli_list.append(("".join(paulis)[::-1], weight))

    return pauli_list

def save_to_json(data, file=None):
    if file==None:
        file="max_cut_paulis/max_cut_paulis.json"
    else:
        file="max_cut_paulis/"+file

    with open(file, "w") as f:
        json.dump(data, f)