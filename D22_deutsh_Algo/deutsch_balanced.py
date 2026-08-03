from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

qc = QuantumCircuit(2,1)

qc.x(1)

qc.h([0,1])

# Balanced Oracle
qc.cx(0,1)

qc.h(0)

qc.measure(0,0)

print(qc)

simulator = AerSimulator()

compiled = transpile(qc, simulator)

job = simulator.run(compiled, shots=1000)

counts = job.result().get_counts()

print(counts)