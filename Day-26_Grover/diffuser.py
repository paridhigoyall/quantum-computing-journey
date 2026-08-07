from qiskit import QuantumCircuit


def diffuser():

    qc = QuantumCircuit(2)

    # Step 1
    qc.h([0, 1])

    # Step 2
    qc.x([0, 1])

    # Step 3
    qc.h(1)
    qc.cx(0, 1)
    qc.h(1)

    # Step 4
    qc.x([0, 1])

    # Step 5
    qc.h([0, 1])

    return qc