from qiskit import QuantumCircuit

def convert_to_circuit(circuit: list) -> QuantumCircuit:
    """
    Converts rustiq circuit to a quantumcircuit
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