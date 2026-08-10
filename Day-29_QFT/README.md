# Day 29 – Quantum Fourier Transform (QFT) Implementation

## 🎯 Objective

Build and experiment with a **3-qubit Quantum Fourier Transform (QFT)** circuit manually using Qiskit.

The goal of today's lesson was to understand how QFT uses **Hadamard gates, controlled phase rotations, and swap gates** to transform quantum phase information.

---

## 🧠 Concepts Covered

- Quantum Fourier Transform
- Phase rotations
- Controlled phase gates
- Hadamard gates
- Relative phase
- Qubit ordering
- Swap gates
- Computational basis
- Measurement and phase information
- QFT circuit construction

---

## 📂 Project Structure

```text
Day-29_QFT/

│── qft_3qubit_manual.py
│── phase_rotations.py
└── README.md
```

---

## 📄 `qft_3qubit_manual.py`

A complete 3-qubit QFT circuit was constructed manually using:

- Hadamard gates
- Controlled phase rotations
- Swap gates
- Qiskit Aer simulation
- 1024 measurement shots

The input state used was:

```text
|101⟩
```

---

## 🔬 QFT Circuit

The main structure of the 3-qubit QFT is:

```text
q0 ──H───CP(π/2)───CP(π/4)────────
q1 ──────H─────────CP(π/2)────────
q2 ────────────────H──────────────
                    │
                 Swap
```

The exact circuit ordering depends on the qubit convention used.

---

## 📐 Phase Rotation Angles

The QFT uses rotation angles determined by:

\[
\theta_k = \frac{2\pi}{2^k}
\]

Therefore:

```text
R₂ = π/2
R₃ = π/4
R₄ = π/8
```

The progressively smaller angles encode increasingly fine phase information.

---

## 🔄 Swap Gates

The QFT naturally produces the output qubits in reversed order.

For example:

```text
Expected:

q0 q1 q2

QFT output:

q2 q1 q0
```

Therefore, swap gates are applied at the end to restore the conventional qubit ordering.

---

## 🧪 Experiment

Input state:

```text
|101⟩
```

Number of shots:

```text
1024
```

Observed measurement results:

```text
101 → 132
000 → 114
010 → 140
011 → 124
110 → 117
001 → 149
111 → 129
100 → 119
```

Theoretical probability for each of the 8 states:

\[
P = \frac{1}{8} = 12.5\%
\]

Expected counts with 1024 shots:

\[
1024 \times \frac18 = 128
\]

The measured counts fluctuate around 128 because of statistical sampling.

---

## 💡 Important Observation

Although the input state was:

```text
|101⟩
```

the output measurement did **not** simply return `101`.

Instead, approximately equal probabilities were observed for all eight computational basis states.

This demonstrates an important property of QFT:

> **The useful information can be encoded in the relative phases rather than directly in the measurement probabilities.**

---

## 🔑 Key Learning

Measurement in the computational basis does not directly reveal relative phase.

For example:

\[
\frac{|0\rangle + |1\rangle}{\sqrt2}
\]

and

\[
\frac{|0\rangle - |1\rangle}{\sqrt2}
\]

have the same computational-basis measurement probabilities:

\[
P(0)=P(1)=\frac12
\]

but they have different relative phases.

Therefore, QFT is normally used as part of a larger quantum algorithm that can convert phase information into measurable information.

---

## 📊 Grover vs QFT

| Grover's Algorithm | Quantum Fourier Transform |

| Search algorithm | Quantum transformation |
| Finds marked states | Reveals phase/periodic structure |
| Oracle + diffuser | Hadamard + controlled phase rotations |
| Amplifies amplitudes | Rearranges phase information |
| Measurement probability is amplified | Phase relationships are transformed |

---

## 🧠 Key Takeaways

- QFT manipulates **phase relationships**.
- Controlled phase gates are fundamental components of QFT.
- Rotation angles become smaller as the required phase precision increases.
- Swap gates reverse the qubit order at the end.
- Measuring immediately after QFT may not reveal the encoded information directly.
- The information is not destroyed; it is represented in the **phase relationships** of the resulting quantum state.
- QFT becomes especially powerful when combined with other algorithms.

---

## 🚀 Applications of QFT

The Quantum Fourier Transform is an important component of:

- Quantum Phase Estimation (QPE)
- Shor's Algorithm
- Period Finding
- Quantum Simulation
- Quantum Signal Processing

---

## 📈 Complexity

Standard QFT implementation:

\[
O(n^2)
\]

where \(n\) is the number of qubits.

Approximate QFT implementations can reduce the number of gates by ignoring very small-angle rotations.

---

## 🎓 What I Learned Today

Today I manually implemented a 3-qubit Quantum Fourier Transform using Qiskit.

I learned that QFT does not simply produce a more obvious measurement result. Instead, it transforms the **phase structure** of a quantum state.

The experiment with the input state `|101⟩` showed approximately equal measurement probabilities across all eight basis states, demonstrating that the useful information cannot always be seen directly from computational-basis measurements.

This helped me understand why QFT is used as a building block for more advanced algorithms.

---

## 🎯 Next Step

### Day 30 – Quantum Phase Estimation (QPE)

Next, I will learn how to extract **phase information and convert it into a measurable value**.

Topics:

- Eigenstates
- Eigenvalues
- Controlled unitary operations
- Phase kickback
- Inverse QFT
- Phase estimation
- QPE implementation in Qiskit

This will provide the next major building block toward understanding **Shor's Algorithm**.
