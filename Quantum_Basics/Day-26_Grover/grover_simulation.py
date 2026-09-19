from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from oracle import oracle
from diffuser import diffuser

qc = QuantumCircuit(2, 2)

qc.h([0, 1])

qc.compose(oracle(), inplace=True)

qc.compose(diffuser(), inplace=True)

qc.measure([0, 1], [0, 1])

sim = AerSimulator()

compiled = transpile(qc, sim)

job = sim.run(compiled, shots=1000)

result = job.result()

counts = result.get_counts()

print(counts)