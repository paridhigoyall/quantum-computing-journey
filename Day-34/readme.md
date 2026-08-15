# Day 34 – Continued Fractions & Shor Period Recovery

## 🎯 Objective

Learn how Shor's Algorithm converts quantum measurement results into a useful candidate for the hidden period.

Today's focus was:

\[
\boxed{
\text{Quantum Measurement}
\rightarrow
\text{Fraction}
\rightarrow
\text{Continued Fractions}
\rightarrow
\text{Candidate Period}
\rightarrow
\text{Validation}
}
\]

This is an important step between the quantum part of Shor's Algorithm and the classical factor-extraction part.

---

## 🧠 Concepts Covered

- Quantum measurement results
- Binary fractions
- Continued fractions
- Continued-fraction convergents
- Candidate period extraction
- Period validation
- Useful vs. unhelpful measurements
- Modular arithmetic
- GCD factor extraction
- Probabilistic nature of Shor's Algorithm

---

## 📂 Project Structure

```text
Day-34/

├── continued_fractions.py
├── period_recovery_cf.py
└── README.md
```

---

## 1. Why Continued Fractions?

In Shor's Algorithm, the quantum computer does not necessarily give us the period directly.

Instead, after the QFT/phase-estimation stage, we obtain a measured value:

\[
\frac{y}{2^n}
\]

Ideally, this value is close to:

\[
\frac{s}{r}
\]

where:

- \(s\) is an integer
- \(r\) is the unknown period

Continued fractions help us find rational approximations and extract possible denominators that can be tested as candidate periods.

---

## 2. Example: Useful Measurement

Suppose we have four counting qubits and measure:

```text
0100
```

Binary `0100` represents:

\[
4
\]

Since four counting qubits represent:

\[
2^4=16
\]

possible values, the measured fraction is:

\[
\frac{4}{16}
=
\frac14
\]

Therefore:

\[
\boxed{\frac14}
\]

The denominator gives the candidate period:

\[
\boxed{r=4}
\]

---

## 3. Continued Fraction

For a fraction such as:

\[
\frac{5}{16}
\]

the continued-fraction representation is:

\[
[0;3,5]
\]

The convergents are:

\[
0,\quad\frac13,\quad\frac5{16}
\]

Their denominators give candidate values:

```text
1
3
16
```

These candidates must then be tested mathematically.

---

## 4. `continued_fractions.py`

The file implements continued-fraction expansion and convergent generation.

For:

\[
\frac5{16}
\]

the program produces:

```text
Fraction: 5/16

Continued fraction:
[0, 3, 5]

Convergents:
0
1/3
5/16
```

---5. Candidate Period Validation

A candidate period \(r\) must satisfy several conditions.

First:

\[
r>0
\]

Second:

\[
r\text{ must be even}
\]

Third:

\[
a^r\bmod N=1
\]

For our example:

\[
N=15,\qquad a=2
\]

the actual period is:

\[
r=4
\]

because:

\[
2^4\bmod15=1
\]

---

## 6. Why \(r=16\) Is Not Useful

For the measurement:

\[
\frac5{16}
\]

one candidate denominator is:

\[
r=16
\]

Interestingly:

\[
2^{16}\bmod15=1
\]

so it technically satisfies the modular condition.

However:

\[
2^{16/2}\bmod15
=
2^8\bmod15
=
1
\]

This produces only trivial GCD results:

\[
\gcd(1-1,15)=15
\]

and:

\[
\gcd(1+1,15)=1
\]

Therefore, \(r=16\) does not provide useful factors.

This demonstrates an important principle:

> A candidate satisfying \(a^r\equiv1\pmod N\) is not automatically a useful period for factor extraction.

---

## 7. Unhelpful Measurement

For:

```text
0101
```

we obtain:

\[
\frac5{16}
\]

The candidate periods generated from the convergents were:

```text
r = 1
r = 3
r = 16
```

After validation:

```text
r = 1   → False
r = 3   → False
r = 16  → False
```

Therefore, this measurement did not provide a useful period.

This is normal.

Shor's Algorithm is probabilistic, so not every measurement produces enough information to recover the period.

---

## 8. Useful Measurement

When we use:

```text
0100
```

we obtain:

\[
\frac14
\]

The continued fraction gives:

\[
\frac14
\]

with candidate:

\[
r=4
\]

Validation:

\[
2^4\bmod15=1
\]

and:

\[
2^{4/2}\bmod15
=
2^2\bmod15
=
4
\]

The value is neither:

\[
1
\]

nor:

\[
-1\bmod15
\]

so the candidate is useful.

---

## 9. Extracting the Factors

With:

\[
r=4
\]

we calculate:

\[
x=2^{r/2}\bmod15
\]

Therefore:

\[
x=4
\]

Then:

\[
\gcd(x-1,15)
=
\gcd(3,15)
=
3
\]

and:

\[
\gcd(x+1,15)
=
\gcd(5,15)
=
5
\]

Therefore:

\[
\boxed{15=3\times5}
\]

---

## 🔄 Complete Period-Recovery Pipeline

Today's implementation can be summarized as:

```text
Quantum Measurement
        ↓
     y / 2ⁿ
        ↓
Continued Fractions
        ↓
   Convergents
        ↓
Candidate Period r
        ↓
Validate r
        ↓
a^(r/2) mod N
        ↓
       GCD
        ↓
     Factors
```

---

## 📊 Example Results

| Measurement | Fraction | Candidate | Result |
|---|---:|---:|---|
| `0100` | \(1/4\) | \(r=4\) | ✅ Useful |
| `1000` | \(1/2\) | \(r=2\) | ❌ Invalid |
| `1100` | \(3/4\) | \(r=4\) | ✅ Useful |
| `0101` | \(5/16\) | \(r=1,3,16\) | ❌ Unhelpful |

---

## 💡 Key Learning

One of the most important lessons from today is:

> **A quantum measurement does not necessarily give the period directly.**

Instead, we obtain a value related to:

\[
\frac{s}{r}
\]

and use classical mathematics to recover possible values of \(r\).

Continued fractions provide a systematic way to find those candidate values.

---

## 🎓 What I Learned Today

Today I learned how the output of the quantum period-finding stage is processed classically.

I learned that:

- Quantum measurements produce fractions related to the hidden period.
- Continued fractions generate rational approximations.
- The denominators of convergents become candidate periods.
- Candidate periods must be mathematically validated.
- Some measurements are useful while others are not.
- Shor's Algorithm may need repeated quantum runs.
- Once a valid period is found, classical GCD calculations can extract the factors.

For the successful measurement:

\[
0100
\rightarrow
\frac14
\rightarrow
r=4
\rightarrow
3,5
\]

Therefore:

\[
\boxed{15=3\times5}
\]

---

## 🚀 Next Step

The next step is to make the implementation more realistic by connecting:

\[
\text{actual Qiskit measurement counts}
\rightarrow
\text{continued fractions}
\rightarrow
\text{candidate periods}
\rightarrow
\text{automatic factor recovery}
\]

We will also explore why some choices of \(a\) work while others fail.

---

## 🏆 Day 34 Milestone

Successfully implemented continued-fraction-based period recovery and demonstrated both:

### Unhelpful measurement

\[
0101\rightarrow\frac5{16}
\rightarrow\text{no useful }r
\]

### Useful measurement

\[
0100\rightarrow\frac14
\rightarrow r=4
\rightarrow 15=3\times5
\]

This completes the **continued-fraction and period-recovery stage of Shor's Algorithm**.
