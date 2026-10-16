
#!/usr/bin/env python3
"""
Complete Joint Weight Enumerator (CJWE) of linear codes over F_q.

Generator matrix input:
    [[1, 0, 1], [0, 1, 1]]

Field elements are encoded by integers 0, ..., q-1.
For q = p^m with m > 1, the encoding represents polynomial
coefficients over F_p modulo an irreducible polynomial.
"""

import sys
import ast
from itertools import product
from collections import Counter


# ============================================================
# FINITE FIELD F_q
# ============================================================

def prime_power(q):
    """Return (p, m) if q = p^m, otherwise None."""
    for p in range(2, q + 1):
        if q % p == 0:
            t = q
            m = 0

            while t % p == 0:
                t //= p
                m += 1

            return (p, m) if t == 1 else None

    return None


def poly_mod(a, f, p):
    """Polynomial remainder; coefficients are low-degree first."""
    a = a[:]
    degree = len(f) - 1

    for i in range(len(a) - 1, degree - 1, -1):
        c = a[i] % p

        if c:
            for j in range(degree + 1):
                a[i - degree + j] = (
                    a[i - degree + j] - c * f[j]
                ) % p

    return [x % p for x in a[:degree]]


def is_irreducible(f, p):
    """Test irreducibility by checking monic divisors."""
    m = len(f) - 1

    for d in range(1, m // 2 + 1):
        for low in product(range(p), repeat=d):
            g = list(low) + [1]

            if not any(poly_mod(f, g, p)):
                return False

    return True


class GF:
    """Finite field with precomputed addition and multiplication tables."""

    def __init__(self, q):
        pm = prime_power(q)

        if pm is None:
            raise ValueError(f"{q} is not a prime power.")

        self.q = q
        self.p, self.m = pm
        p, m = self.p, self.m
        self.modpoly = None

        if m == 1:
            self.add = [
                [(a + b) % q for b in range(q)]
                for a in range(q)
            ]

            self.mul = [
                [(a * b) % q for b in range(q)]
                for a in range(q)
            ]

        else:
            for low in product(range(p), repeat=m):
                f = list(low) + [1]

                if f[0] != 0 and is_irreducible(f, p):
                    self.modpoly = f
                    break

            if self.modpoly is None:
                raise ValueError("Could not construct the finite field.")

            f = self.modpoly

            def digits(x):
                return [(x // (p ** i)) % p for i in range(m)]

            def encode(coeffs):
                return sum(
                    c * (p ** i)
                    for i, c in enumerate(coeffs)
                )

            representations = [digits(a) for a in range(q)]

            # Addition table.
            self.add = [
                [
                    encode([
                        (x + y) % p
                        for x, y in zip(
                            representations[a],
                            representations[b]
                        )
                    ])
                    for b in range(q)
                ]
                for a in range(q)
            ]

            # Multiplication table.
            def multiply_poly(a, b):
                A = representations[a]
                B = representations[b]
                result = [0] * (2 * m - 1)

                for i, x in enumerate(A):
                    if x == 0:
                        continue

                    for j, y in enumerate(B):
                        if y:
                            result[i + j] = (
                                result[i + j] + x * y
                            ) % p

                return encode(poly_mod(result, f, p))

            self.mul = [
                [
                    multiply_poly(a, b)
                    for b in range(q)
                ]
                for a in range(q)
            ]

        # Additive inverses.
        self.neg = [
            next(b for b in range(q) if self.add[a][b] == 0)
            for a in range(q)
        ]

        # Multiplicative inverses; inverse of zero is undefined.
        self.inv = [None] * q

        for a in range(1, q):
            self.inv[a] = next(
                b for b in range(1, q)
                if self.mul[a][b] == 1
            )

    def describe(self):
        if self.m == 1:
            return f"F_{self.q} = integers modulo {self.q}"

        terms = " + ".join(
            f"{c}*t^{i}" if i else str(c)
            for i, c in enumerate(self.modpoly)
            if c
        )

        return (
            f"F_{self.q} = F_{self.p}[t]/({terms}). "
            f"Elements are encoded by integers 0,...,{self.q - 1}."
        )


# ============================================================
# INPUT HELPERS
# ============================================================

def ask_int(prompt, lo=None, hi=None):
    while True:
        try:
            value = int(input(prompt).strip())

            if (
                (lo is None or value >= lo)
                and (hi is None or value <= hi)
            ):
                return value

        except ValueError:
            pass

        print("Invalid input. Please try again.")


def ask_field():
    while True:
        q = ask_int(
            "Field size q (e.g. 2, 3, 4, 5, 7, 8, 9): ",
            2
        )

        try:
            F = GF(q)
            print(" ", F.describe())
            return F

        except ValueError as error:
            print(" ", error)


def ask_generator_matrix(F, n, label):
    q = F.q

    k = ask_int(
        f"\n[{label}] Number of rows k: ",
        1
    )

    print("\nGenerator matrix input format")
    print("-----------------------------")
    print(f"Enter a list containing exactly {k} rows.")
    print(f"Each row must contain exactly {n} entries.")
    print(f"Entries must be integers from 0 to {q - 1}.")
    print("\nExample:")
    print("  [[1, 0, 1], [0, 1, 1]]")
    print("\nEnter the entire matrix on one line.")

    while True:
        try:
            G = ast.literal_eval(
                input(f"\n[{label}] Enter G: ").strip()
            )

            if not isinstance(G, list) or len(G) != k:
                raise ValueError(
                    f"G must contain exactly {k} rows."
                )

            for i, row in enumerate(G):
                if not isinstance(row, list) or len(row) != n:
                    raise ValueError(
                        f"Row {i + 1} must contain exactly {n} entries."
                    )

                if any(
                    type(x) is not int or not 0 <= x < q
                    for x in row
                ):
                    raise ValueError(
                        f"Every entry must be an integer "
                        f"from 0 to {q - 1}."
                    )

            print("\nAccepted generator matrix:")
            for row in G:
                print(" ", row)

            return G

        except (ValueError, SyntaxError, TypeError) as error:
            print("Invalid matrix:", error)
            print("Example: [[1, 0, 1], [0, 1, 1]]")


# ============================================================
# GAUSSIAN ELIMINATION OVER F_q
# ============================================================

def row_basis(F, G, n):
    """
    Return independent rows spanning the same code as G.

    Uses Gaussian elimination over the actual finite field.
    """
    A = [row[:] for row in G]
    rank = 0

    for col in range(n):
        pivot = next(
            (
                i for i in range(rank, len(A))
                if A[i][col] != 0
            ),
            None
        )

        if pivot is None:
            continue

        A[rank], A[pivot] = A[pivot], A[rank]

        # Normalize pivot to 1.
        inverse = F.inv[A[rank][col]]

        A[rank] = [
            F.mul[inverse][x]
            for x in A[rank]
        ]

        pivot_row = A[rank]

        # Eliminate entries below the pivot.
        for i in range(rank + 1, len(A)):
            factor = A[i][col]

            if factor == 0:
                continue

            negative_factor = F.neg[factor]

            for j in range(col, n):
                A[i][j] = F.add[A[i][j]][
                    F.mul[negative_factor][pivot_row[j]]
                ]

        rank += 1

        if rank == len(A):
            break

    return A[:rank]


# ============================================================
# CODEWORD GENERATION
# ============================================================

def span(F, basis, n):
    """
    Generate each codeword exactly once from an independent basis.

    Number of codewords = q^rank.
    """
    codewords = [tuple([0] * n)]

    for row in basis:
        old_words = codewords

        # Precompute scalar multiples of the current basis row.
        scaled_rows = [
            tuple(F.mul[c][x] for x in row)
            for c in range(1, F.q)
        ]

        new_words = []

        for word in old_words:
            for scaled in scaled_rows:
                new_words.append(
                    tuple(
                        F.add[word[j]][scaled[j]]
                        for j in range(n)
                    )
                )

        codewords = old_words + new_words

    return codewords


# ============================================================
# COMPLETE JOINT WEIGHT ENUMERATOR
# ============================================================

def cjwe(F, c1, c2):
    """
    Compute CJWE(C1, C2).

    The exponent of x[a,b] counts positions where u_i=a, v_i=b.
    """
    q = F.q
    q_squared = q * q
    cnt = Counter()

    # Cache flattened pair-symbol indices for faster counting.
    # Each pair (a,b) is represented by a*q + b.
    for u in c1:
        for v in c2:
            counts = [0] * q_squared

            for a, b in zip(u, v):
                counts[a * q + b] += 1

            cnt[tuple(counts)] += 1

    return cnt


# ============================================================
# POLYNOMIAL OUTPUT
# ============================================================

def mono(variables, exponents):
    parts = []

    for variable, exponent in zip(variables, exponents):
        if exponent == 1:
            parts.append(variable)
        elif exponent > 1:
            parts.append(f"{variable}^{exponent}")

    return "*".join(parts) if parts else "1"


def show_poly(table, variables, title):
    print(f"\n{title}")
    print("-" * len(title))

    terms = []

    for exponents in sorted(table, reverse=True):
        coefficient = table[exponents]

        if coefficient == 0:
            continue

        monomial = mono(variables, exponents)

        if monomial == "1":
            terms.append(str(coefficient))
        else:
            terms.append(f"{coefficient}*{monomial}")

    print(" + ".join(terms) if terms else "0")
    print(f"\nNumber of terms = {len(terms)}")
    print(f"Sum of coefficients = {sum(table.values())}")


# ============================================================
# MAIN PROGRAM
# ============================================================

def program_cjwe():
    print("\n=== Complete Joint Weight Enumerator ===")

    F = ask_field()
    n = ask_int("Common length n: ", 1)

    num = ask_int(
        "Number of codes (at least 2; all pairs are processed): ",
        2
    )

    codes = []

    for i in range(1, num + 1):
        G = ask_generator_matrix(F, n, f"code {i}")

        basis = row_basis(F, G, n)
        rank = len(basis)

        code = span(F, basis, n)
        codes.append(code)

        print(f"\nCode C{i}:")
        print(f"  Rank/dimension = {rank}")
        print(f"  Number of codewords = {len(code)}")
        print(f"  Expected size q^rank = {F.q ** rank}")

    q = F.q

    variables = [
        f"x[{a},{b}]"
        for a in range(q)
        for b in range(q)
    ]

    for i in range(num):
        for j in range(i + 1, num):
            result = cjwe(F, codes[i], codes[j])

            show_poly(
                result,
                variables,
                f"CJWE of (C{i + 1}, C{j + 1}) "
                f"(x[a,b] counts positions with symbols a,b)"
            )

            expected_sum = len(codes[i]) * len(codes[j])

            if sum(result.values()) != expected_sum:
                raise RuntimeError(
                    "CJWE verification failed: coefficient sum is incorrect."
                )


if __name__ == "__main__":
    try:
        program_cjwe()
    except (EOFError, KeyboardInterrupt):
        print("\nProgram terminated.")
        sys.exit(0)
