# Day 35 – Quantum Amplitude Estimation

## 🎯 Objective

Learn how Quantum Amplitude Estimation (QAE) uses the ideas of:

- Amplitude encoding
- Grover-style rotations
- Phase estimation
- QFT / inverse QFT

to estimate an unknown probability.

The main relationship studied today was:

\[
\boxed{
a
\rightarrow
\theta
\rightarrow
2\theta
\rightarrow
\phi
\rightarrow
\text{QPE}
\rightarrow
a
}
\]

---

## 🧠 Core Concept

Suppose a quantum state is:

\[|\psi\rangle=
\sqrt{1-a}|bad\rangle
+
\sqrt{a}|good\rangle
\]

where \(a\) is the probability of measuring a good state.

The amplitude of the good state is:

\[
\sqrt{a}
\]

and therefore:

\[
a=(\sqrt a)^2
\]

We introduce an angle \(\theta\):

\[
\sqrt{a}=\sin\theta
\]

Therefore:

\[
\boxed{a=\sin^2\theta}
\]

---

## 📌 Example Used Today

We used:

\[
a=0.25
\]

Therefore:

\[
\sqrt a=\frac12
\]

and:

\[
\sin\theta=\frac12
\]

so:

\[
\boxed{\theta=\frac{\pi}{6}}
\]

The state becomes:

\[
|\psi\rangle=
\frac{\sqrt3}{2}|0\rangle
+
\frac12|1\rangle
\]

---

## 💻 Basic Amplitude Preparation

The state was prepared using:

\[
R_y(2\theta)
\]

Since:

\[
2\theta=\frac{\pi}{3}
\]

we used:

```python
qc.ry(np.pi / 3, 0)
