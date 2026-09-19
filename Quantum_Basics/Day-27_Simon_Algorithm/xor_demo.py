def xor(a, b):
    return ''.join(
        str(int(x) ^ int(y))
        for x, y in zip(a, b)
    )

secret = "101"

tests = [
    "000",
    "001",
    "010",
    "011"
]

for t in tests:
    print(f"{t} XOR {secret} = {xor(t, secret)}")