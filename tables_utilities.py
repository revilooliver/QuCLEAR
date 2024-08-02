import json
import os
from typing import Dict, Tuple

def get_results(file_dir: str, tag: str, experiment:str) -> Dict:
    """For all results for a given results tag and experiment.
    Args: 
        file_dir: dir
        tag: unique file identifier: quclear_tag="new"
                                    quclear_tag2="our"
                                    qiskit_tag="qiskit"
                                    rustiq_tag="rustiq"
                                    paulihedral_tag="paulihedral"
                                    pytket_tag="pytket" 
        experiment: experiment identifier: 
                    "Paulis"| "H2O"| "LiH" | "benzene"|"max_cut"|"labs"|"""
    # Filter files
    all_files = os.listdir(file_dir)
    all_files = [f for f in all_files if experiment in f]
    if not (tag=="qiskit" or tag=="rustiq"):
        new_all_files=[f for f in all_files if tag in f]
    else:
        new_all_files=[]
        for file_name in all_files:
            with open(file_dir+file_name) as f:
                if tag in f.read():
                    new_all_files.append(file_name)
        # print(tag)
        # print(new_all_files)

    results={}
    for file_name in new_all_files:
        with open(file_dir+ file_name, "r") as f:
            data=json.load(f)
            num_qubits=get_num_qubits(data[0])
            data[1][0]["num_qubits"]=num_qubits
            results[file_name]=data[1][0]
    return results

def get_num_qubits(data):
    '''Gets num qubits for experiment.'''
    for elem in data: #flat list of hamiltonians
        if isinstance(elem, str):
            # print(elem)
            return len(elem)
        else: # Hamiltonian in list of blocks
            # print(elem[0])
            return len(elem[0])

def filter_dict_val_for_tag(data, tag):
    # print(data)
    # print(tag)
    # The rounding is for the time value.
    return [round(v,4) for k, v in data.items() if tag in k][0]

def get_metrics(data,tag):
    results={}
    for k, v in data.items():
        # print(k)
        # print(v)
        paulis=v["num_paulis"]
        times=filter_dict_val_for_tag(v["times"], tag)
        cnots=filter_dict_val_for_tag(v["gate_counts"], tag)
        depth=filter_dict_val_for_tag(v["circuit_entangling_depth"], tag)
        results[k]={"num_paulis": paulis, "times": times, "cnots": cnots, "circuit_entangling_depth": depth, "num_qubits": v["num_qubits"]}
    return results

def test_lenghts_res(data):
    # test
    print(all([len(elem) for block in data for elem in block]))


def get_data_handles(quclear, qiskit, rustiq, paulih, pytket, file) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    '''Gets handles for specific file. Returns in the specific order:
    quclear, qiskit, rustiq, paulihedral, pytket'''
    quclear_f=quclear[file]
    qiskit_f=qiskit[file.replace("_new_", "_")]
    rustiq_f=rustiq[file.replace("_new_", "_")]
    paulih_f=paulih[file.replace("_new_", "_paulihedral_")]
    pytket_f=pytket[file.replace("_new_", "_pytket_")]
    return quclear_f, qiskit_f, rustiq_f, paulih_f, pytket_f