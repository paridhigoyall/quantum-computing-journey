from fractions import Fraction
from math import gcd


# ==========================================
# Shor parameters
# ==========================================

N = 15
a = 2
counting_qubits = 4


# ==========================================
# Measurement results from quantum circuit
# ==========================================

counts = {
    "0000": 248,
    "0100": 249,
    "1000": 262,
    "1100": 265
}


# ==========================================
# Convert measurement to fraction
# ==========================================

def measurement_to_fraction(bitstring):
    value = int(bitstring, 2)

    return Fraction(
        value,
        2 ** counting_qubits
    )


# ==========================================
# Validate candidate period
# ==========================================

def validate_period(a, N, r):

    # Period must be positive
    if r <= 0:
        return False

    # Period must be even
    if r % 2 != 0:
        return False

    # Check a^r mod N = 1
    if pow(a, r, N) != 1:
        return False

    # Calculate a^(r/2)
    x = pow(a, r // 2, N)

    # Must not be -1 mod N
    if x == N - 1:
        return False

    return True


# ==========================================
# Extract factors
# ==========================================

def extract_factors(a, N, r):

    x = pow(a, r // 2, N)

    factor1 = gcd(x - 1, N)
    factor2 = gcd(x + 1, N)

    return factor1, factor2


# ==========================================
# Process measurements
# ==========================================

print("Shor Period Recovery")
print("=" * 40)

for bitstring, shots in sorted(
    counts.items(),
    key=lambda item: item[1],
    reverse=True
):

    fraction = measurement_to_fraction(bitstring)

    candidate_r = fraction.denominator

    valid = validate_period(
        a,
        N,
        candidate_r
    )

    print(
        f"\nMeasurement : {bitstring}"
        f"\nShots       : {shots}"
        f"\nFraction    : {fraction}"
        f"\nCandidate r : {candidate_r}"
        f"\nValid       : {valid}"
    )

    if valid:

        factor1, factor2 = extract_factors(
            a,
            N,
            candidate_r
        )

        print(
            f"Factors     : {factor1} × {factor2}"
        )

        if (
            factor1 != 1
            and factor2 != 1
            and factor1 != N
            and factor2 != N
        ):
            print("\nSUCCESS!")
            print(
                f"{N} = {factor1} × {factor2}"
            )

            break