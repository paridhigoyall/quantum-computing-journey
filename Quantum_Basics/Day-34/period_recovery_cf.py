from fractions import Fraction
from math import gcd


N = 15
a = 2
counting_qubits = 4


def continued_fraction_convergents(fraction):
    numerator = fraction.numerator
    denominator = fraction.denominator

    coefficients = []

    while denominator != 0:
        quotient = numerator // denominator
        coefficients.append(quotient)

        numerator, denominator = (
            denominator,
            numerator - quotient * denominator
        )

    convergents = []

    p_minus_2, p_minus_1 = 0, 1
    q_minus_2, q_minus_1 = 1, 0

    for coefficient in coefficients:

        p = coefficient * p_minus_1 + p_minus_2
        q = coefficient * q_minus_1 + q_minus_2

        convergents.append(Fraction(p, q))

        p_minus_2, p_minus_1 = p_minus_1, p
        q_minus_2, q_minus_1 = q_minus_1, q

    return convergents


def validate_period(r):

    if r <= 0:
        return False

    # r must be even
    if r % 2 != 0:
        return False

    # a^r mod N must equal 1
    if pow(a, r, N) != 1:
        return False

    # Check that r gives non-trivial factors
    x = pow(a, r // 2, N)

    if x == 1 or x == N - 1:
        return False

    factor1 = gcd(x - 1, N)
    factor2 = gcd(x + 1, N)

    # Both factors must be non-trivial
    if factor1 in (1, N) or factor2 in (1, N):
        return False

    return True 

# --------------------------------
# Quantum measurement
# --------------------------------

bitstring = "0101"

value = int(bitstring, 2)

fraction = Fraction(
    value,
    2 ** counting_qubits
)

print("Measurement:", bitstring)
print("Fraction:", fraction)

# --------------------------------
# Continued fractions
# --------------------------------

convergents = continued_fraction_convergents(
    fraction
)

print("\nConvergents:")

for convergent in convergents:

    r = convergent.denominator

    valid = validate_period(r)

    print(
        f"{convergent} "
        f"→ candidate r = {r} "
        f"→ valid = {valid}"
    )