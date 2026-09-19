# Day 32 – Shor's Algorithm: Quantum Period Finding

## 🎯 Objective

Understand the core idea behind **Shor's Algorithm** and implement an educational version for factoring:

[
N=15
]

The main focus of today's lesson was understanding how quantum period finding, the Quantum Fourier Transform (QFT), and classical number theory work together to find the factors of an integer.

---

## 🧠 Concepts Covered

* Shor's Algorithm
* Period finding
* Modular arithmetic
* Modular exponentiation
* Quantum period finding
* Quantum Fourier Transform
* Phase information
* Inverse QFT
* GCD-based factor extraction
* Classical and quantum parts of Shor's Algorithm

---

## 📂 Project Structure

```text
Day-32_Shors/

│── period_finding.py
│── shors_15.py
└── README.md
```

---

## 1. The Problem

For today's example:

[
N=15
]

The goal is to find its non-trivial factors:

[
15=3\times5
]

For small numbers, classical factorization is easy.

The importance of Shor's Algorithm is that it provides a quantum approach that can factor very large integers much more efficiently than known classical factoring methods.

---

### 2. Shor's Main Idea

Shor's Algorithm does not directly search for the factors.

Instead, it converts the factoring problem into a **period-finding problem**.

We choose:

[
a=2
]

and evaluate:

[
f(x)=2^x\bmod15
]

The resulting sequence is:

```text
x       : 0  1  2  3  4  5  6  7
f(x)    : 1  2  4  8  1  2  4  8
```

The sequence repeats every 4 values.

Therefore:

[
\boxed{r=4}
]

where (r) is the period.

---

### 3. Classical Period Finding

The `period_finding.py` program demonstrates the classical mathematical part.

Example output:

```text
N = 15
a = 2
Period r = 4
a^(r/2) mod N = 4
Factor 1 = 3
Factor 2 = 5
```

---

### 4. From Period to Factors

Once we know:
 [r=4]
we calculate:
[a^{r/2}\bmod N
]
Therefore:
[
2^{4/2}\bmod15 = 2^2\bmod15 = 4
]

Let:

[
x=4
]

We then calculate:

[
\gcd(x-1,N)
]

and:

[
\gcd(x+1,N)
]

Therefore:

[
\gcd(3,15)=3
]

and:

[
\gcd(5,15)=5
]

Giving:

[
\boxed{15=3\times5}
]

---

## 5. Quantum Period Finding

The important quantum part is implemented in `shors_15.py`.

The circuit uses:

* 4 counting qubits
* 4 target qubits
* Hadamard gates
* Controlled modular multiplication
* Inverse QFT
* Measurement

The quantum state represents the modular relationship:

[
|x\rangle|1\rangle
\rightarrow
|x\rangle|2^x\bmod15\rangle
]

This creates the periodic structure inside the quantum state.

---

## 6. Modular Pattern

For:

[
a=2,\quad N=15
]

we have:

```text
2^0 mod 15 = 1
2^1 mod 15 = 2
2^2 mod 15 = 4
2^3 mod 15 = 8
2^4 mod 15 = 1
```

Therefore:

```text
1 → 2 → 4 → 8 → 1 → ...
```

and:

[
\boxed{r=4}
]

---

## 7. Quantum Fourier Transform

The inverse QFT is applied to the counting register.

Since the period is:

[
r=4
]

the Fourier peaks occur approximately at:

[
0,\frac14,\frac12,\frac34
]

With four counting qubits:

[
2^4=16
]

possible states exist.

The important measured states were:

```text
0000
0100
1000
1100
```

---

## 8. Simulation Results

The quantum simulation produced:

```text
0000: 248
0100: 249
1000: 262
1100: 265
```

The four peaks are close to one another, as expected from the periodic structure.

---

## 9. Interpreting `0100`

For a 4-qubit counting register:

```text
0100₂ = 4
```

Since the register represents values from:

[
0\text{ to }15
]

the measured value corresponds to:

[
\frac{4}{16}=\frac14
]

Therefore:

[
\boxed{\frac14}
]

The denominator gives the period candidate:

[
\boxed{r=4}
]

In a general implementation, the measured fraction is processed using **continued fractions** to recover a likely period.

---

## 🔗 Complete Shor Pipeline

Today's implementation demonstrated the complete conceptual flow:

```text
Choose a
   ↓
f(x) = a^x mod N
   ↓
Periodic structure
   ↓
Quantum modular exponentiation
   ↓
Inverse QFT
   ↓
Measurement
   ↓
Estimate period r
   ↓
a^(r/2) mod N
   ↓
GCD calculations
   ↓
Factors
```

For today's example:

```text
N = 15
a = 2
r = 4
```

Then:

```text
2^(4/2) mod 15 = 4

gcd(4 - 1, 15) = 3
gcd(4 + 1, 15) = 5
```

Therefore:

[
\boxed{15=3\times5}
]

---

## 📊 Classical vs Quantum Components

| Component               | Type                    |
| ----------------------- | ----------------------- |
| Choosing (a)            | Classical               |
| Modular exponentiation  | Quantum circuit in Shor |
| Period finding          | Quantum                 |
| Inverse QFT             | Quantum                 |
| Measurement             | Quantum                 |
| Continued fractions     | Classical               |
| GCD calculation         | Classical               |
| Final factor extraction | Classical               |

---

## ⚠️ Important Implementation Note

The `UnitaryGate` approach used in this educational implementation represents modular multiplication as a complete unitary matrix.

This is useful for learning because it lets us see the algorithmic structure clearly.

It is **not an efficient hardware-level implementation** of modular arithmetic.

A practical quantum implementation would decompose modular multiplication into elementary quantum gates and would require significantly more sophisticated circuit construction.

---

## 💡 Key Learnings

* Shor's Algorithm transforms factoring into period finding.
* The function (a^x\bmod N) contains the hidden periodic structure.
* Quantum modular exponentiation encodes that structure into a quantum state.
* The inverse QFT reveals frequency information related to the period.
* Measurement produces fractions related to the period.
* Classical number theory then converts the period into factors.
* QFT and QPE concepts learned previously are directly connected to Shor's Algorithm.

---

## 🎓 What I Learned Today

Today I connected several previously learned quantum concepts into one complete algorithmic pipeline.

I learned that Shor's Algorithm does not directly search for factors. Instead, it uses quantum computation to find the period of a modular exponential function.

For:

[
N=15,\quad a=2
]

the period was:

[
r=4
]

The quantum simulation produced Fourier peaks corresponding to:

[
0,\frac14,\frac12,\frac34
]

From the period (r=4), the classical GCD calculations produced:

[
\boxed{3\text{ and }5}
]

Thus:

[
\boxed{15=3\times5}
]

---

## 🚀 Next Step

Next, the goal is to move beyond the educational (N=15) example and understand:

* Continued fractions
* Why measurement gives approximate fractions
* Why some choices of (a) fail
* How Shor handles unsuccessful runs
* More realistic modular arithmetic circuits
* Limitations of simulating Shor on a classical computer
* Running quantum circuits on real quantum hardware

---

## 🏆 Day 32 Milestone

Successfully implemented an educational quantum period-finding circuit for:

[
\boxed{N=15}
]

and observed the expected Fourier peaks:

```text
0000
0100
1000
1100
```

leading to the period:

[
\boxed{r=4}
]

and finally the factors:

[
\boxed{15=3\times5}
]

This completes the first practical implementation of the core idea behind **Shor's Algorithm**.
