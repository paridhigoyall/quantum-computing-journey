from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


def superdense(message):

    qc = QuantumCircuit(2,2)


    # Create Bell pair
    qc.h(0)
    qc.cx(0,1)


    # Encoding
    if message == "00":
        pass

    elif message == "01":
        qc.x(0)

    elif message == "10":
        qc.z(0)

    elif message == "11":
        qc.z(0)
        qc.x(0)


    # Decoding
    qc.cx(0,1)
    qc.h(0)


    # Measurement
    qc.measure([0,1],[0,1])


    return qc



messages = ["00","01","10","11"]


simulator = AerSimulator()


results = {}


for msg in messages:

    circuit = superdense(msg)

    compiled = transpile(circuit, simulator)

    job = simulator.run(compiled, shots=1000)

    counts = job.result().get_counts()

    results[msg] = counts

    print(
        f"Message sent: {msg}"
    )

    print(counts)
    print("----------------")

plot_histogram(results["11"])
plt.show()