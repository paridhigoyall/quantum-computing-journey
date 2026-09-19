from fractions import Fraction


def continued_fraction_convergents(fraction):
    """
    Generate the continued-fraction coefficients
    and their convergents.
    """

    numerator = fraction.numerator
    denominator = fraction.denominator

    coefficients = []

    # --------------------------------
    # Continued-fraction expansion
    # --------------------------------

    while denominator != 0:
        quotient = numerator // denominator
        coefficients.append(quotient)

        numerator, denominator = (
            denominator,
            numerator - quotient * denominator
        )

    # --------------------------------
    # Generate convergents
    # --------------------------------

    convergents = []

    p_minus_2, p_minus_1 = 0, 1
    q_minus_2, q_minus_1 = 1, 0

    for coefficient in coefficients:

        p = coefficient * p_minus_1 + p_minus_2
        q = coefficient * q_minus_1 + q_minus_2

        convergent = Fraction(p, q)
        convergents.append(convergent)

        p_minus_2, p_minus_1 = p_minus_1, p
        q_minus_2, q_minus_1 = q_minus_1, q

    return coefficients, convergents


# --------------------------------
# Test
# --------------------------------

fraction = Fraction(5, 16)

coefficients, convergents = continued_fraction_convergents(
    fraction
)

print("Fraction:", fraction)

print("Continued fraction:")
print(coefficients)

print("\nConvergents:")

for value in convergents:
    print(value)