# Day 28 – Introduction to Quantum Fourier Transform (QFT)

## 📌 Objective

Learn the fundamentals of the Quantum Fourier Transform (QFT), one of the most important subroutines in quantum computing and the core component behind Shor's Algorithm.

---

## 📖 Theory

The Quantum Fourier Transform is the quantum equivalent of the Classical Fourier Transform.

Instead of decomposing classical signals into frequencies, the QFT transforms quantum states into the **phase domain**, allowing hidden periodic patterns to be detected efficiently.

Unlike Grover's Algorithm, which amplifies amplitudes, the QFT manipulates the **relative phases** of quantum states.

---

## 🧠 Key Concepts Learned

- Classical Fourier Transform vs Quantum Fourier Transform
- Phase representation of quantum states
- Hadamard Gate
- Phase Rotation Gates (S, T, Controlled Phase)
- Swap Gates
- Phase Manipulation
- Computational Basis vs Phase Basis

---

## 📂 Files

### `hadamard_review.py`

Reviews the Hadamard gate used at the beginning of every QFT circuit.

---

### `phase_gate_demo.py`

Demonstrates the effect of various phase gates.

Gates used:

- H Gate
- S Gate (π/2 Phase Rotation)
- T Gate (π/4 Phase Rotation)
- Z Gate (π Phase Flip)

---

### `qft_basic.py`

Implements a simple two-qubit Quantum Fourier Transform circuit using:

- Hadamard Gate
- Controlled Phase Gate
- Swap Gate

---

## 🔬 Observations

- Hadamard creates superposition.
- Phase gates modify the relative phase without changing measurement probabilities immediately.
- Controlled Phase Gates encode phase relationships between qubits.
- Swap Gates reverse the output qubit order because the QFT naturally produces the output in reverse order.

---

## 📊 Comparison

| Grover's Algorithm | Quantum Fourier Transform |

|--------------------|---------------------------|
| Searches for marked states | Detects hidden periodicity |
| Amplifies amplitudes | Manipulates phases |
| Uses Oracle + Diffuser | Uses Hadamard + Controlled Phase Gates |
| Search Algorithm | Quantum Transformation |

---

## 🚀 Key Takeaways

- QFT is **not** a search algorithm.
- QFT changes the representation of quantum information into the phase domain.
- Phase information is essential for algorithms like:
  - Quantum Phase Estimation (QPE)
  - Shor's Algorithm
  - Quantum Signal Processing

---

## 💡 What I Learned Today

- Why QFT relies on phase rotation gates instead of CNOT gates.
- Why Swap gates are required at the end of the circuit.
- The difference between amplitude manipulation (Grover) and phase manipulation (QFT).
- QFT serves as a reusable building block rather than a standalone algorithm.

---

## 📈 Complexity

Classical Discrete Fourier Transform:
O(N²)

Fast Fourier Transform (FFT):
O(N log N)

Quantum Fourier Transform:
O(n²) gates for **n qubits** (with the standard implementation)

---

## 🎯 Next Step

Implement the complete multi-qubit Quantum Fourier Transform manually, understand controlled phase rotation angles, and compare it with Qiskit's built-in QFT before moving on to **Quantum Phase Estimation (QPE)**.
