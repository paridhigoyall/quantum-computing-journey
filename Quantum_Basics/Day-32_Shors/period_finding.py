def find_period(a, N):
    """Find the smallest r such that a^r mod N == 1."""
    
    value = 1

    for r in range(1, N):
        value = (value * a) % N

        if value == 1:
            return r

    return None


# Example: factor N = 15
N = 15
a = 2

r = find_period(a, N)

print(f"N = {N}")
print(f"a = {a}")
print(f"Period r = {r}")

if r is not None:
    print(f"a^(r/2) mod N = {pow(a, r // 2, N)}")


from math import gcd

x = pow(a, r // 2, N)

factor1 = gcd(x - 1, N)
factor2 = gcd(x + 1, N)

print(f"Factor 1 = {factor1}")
print(f"Factor 2 = {factor2}")