# Day 30 – Quantum Phase Estimation (QPE)

## 🎯 Objective

Learn the fundamentals of **Quantum Phase Estimation (QPE)** and understand how quantum phase information can be converted into a measurable classical value.

The goal is to estimate the phase `θ` in:

\[
U|\psi\rangle = e^{2\pi i\theta}|\psi\rangle
\]

For today's experiment:

\[
\theta = \frac{1}{2}
\]

---

## 🧠 Concepts Covered

- Quantum Phase Estimation
- Eigenstates
- Eigenvalues
- Phase Kickback
- Controlled Unitary Operations
- Controlled-Z Gate
- Inverse Quantum Fourier Transform
- Measurement
- Phase-to-classical conversion

---

## 📂 Project Structure

```text
Day-30_QPE/

│── phase_kickback.py
│── qpe_basic.py
│── qpe_simulation.py
└── README.md
```

---

## 📄 File Description

### 1. `phase_kickback.py`

Demonstrates the basic idea of **phase kickback**.

The target qubit is prepared in:

```text
|1⟩
```

A Hadamard gate puts the control qubit into superposition, followed by a controlled-Z operation.

Since:

\[
Z|1\rangle = -|1\rangle
\]

the phase is transferred to the control qubit.

---

### 2. `qpe_basic.py`

Builds a simple one-qubit Quantum Phase Estimation circuit.

The circuit contains:

1. Target eigenstate preparation
2. Hadamard on the phase/control qubit
3. Controlled-Z
4. Inverse QFT
5. Measurement

---

### 3. `qpe_simulation.py`

Simulates the QPE circuit using **Qiskit Aer** with 1000 shots.

The estimated phase is:

\[
\theta = \frac{1}{2}
\]

The expected measurement is:

```text
{'1': approximately 1000}
```

---

 Phase Kickback

The target qubit is prepared as:

\[
|1\rangle
\]

Applying the Z gate gives:

\[
Z|1\rangle=-|1\rangle
\]

Since:

\[
-1=e^{i\pi}
\]

and QPE represents the phase as:

\[
e^{2\pi i\theta}
\]

we get:

\[
e^{2\pi i\theta}=e^{i\pi}
\]

Therefore:

\[
2\pi\theta=\pi
\]

and:

\[
\boxed{\theta=\frac12}
\]

---

 QPE Circuit

The basic one-qubit QPE circuit is:

```text
Control ──H────●────H────Measure
               │
Target  ──X────Z──────────────
```

The controlled-Z operation creates the phase kickback.

The final Hadamard acts as the **inverse QFT** for a one-qubit phase register.

---

 Expected Result

For:

\[
\theta=\frac12
\]

the binary representation is:

```text
0.1₂
```

Therefore, the phase qubit should be measured as:

```text
1
```

With 1000 shots, the simulation should produce approximately:

```text
{'1': 1000}
```

---

 Key Learning

QPE follows the general process:

```text
Eigenstate
    ↓
Controlled-U
    ↓
Phase Kickback
    ↓
Inverse QFT
    ↓
Measurement
    ↓
Classical Phase Estimate
```

The important idea is:

> **Phase information cannot be directly read from a computational-basis measurement. QPE uses phase kickback and the inverse QFT to convert the phase into measurable information.**

---

Connection to QFT

In Day 29, the Quantum Fourier Transform showed how quantum phase information can be transformed.

In QPE, we use the **inverse QFT** to decode phase information.

```text
QFT:
Phase information → Fourier representation

QPE:
Phase information → Inverse QFT → Measurable bit string
```

---

 Grover vs QFT vs QPE

| Algorithm | Main Purpose |

| Grover | Search for a marked state |
| QFT | Transform phase/frequency information |
| QPE | Estimate an unknown quantum phase |

---

 What I Learned Today

Today I learned how Quantum Phase Estimation extracts an unknown phase from a quantum operation.

I learned that:

- An eigenstate is required for clean phase estimation.
- Controlled operations create phase kickback.
- The phase can be represented as \(e^{2\pi i\theta}\).
- The inverse QFT converts phase information into a measurable form.
- Measurement then produces the classical estimate of the phase.

For today's example:

\[
\boxed{\theta=\frac12}
\]

and the measured phase bit is:

```text
1
```

---

 Applications

Quantum Phase Estimation is an important component of:

- Shor's Algorithm
- Quantum Chemistry
- Quantum Simulation
- Eigenvalue Estimation
- Hamiltonian Simulation
- Quantum Algorithms for Scientific Computing

---

 Next Step

Day 31 – Multi-Qubit Quantum Phase Estimation

Next, I will increase the phase register from **1 qubit to multiple qubits**.

Topics:

- 2-qubit phase estimation
- Fractional binary phases
- Controlled-\(U^2\), \(U^4\), etc.
- Inverse QFT
- Phase precision
- Measurement probabilities

The goal will be to estimate phases such as:

\[
\frac14,\quad \frac38,\quad \frac34
\]

This will bring us one major step closer to understanding **Shor's Algorithm**.
