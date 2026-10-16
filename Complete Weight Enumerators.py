
import sys
import ast
from itertools import product
from collections import Counter
from math import log


# ============================================================
# FINITE FIELD F_q
# ============================================================

def prime_power(q):
    """Return (p, m) if q = p^m, where p is prime; else None."""
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
    """Polynomial remainder, coefficients ordered from low to high."""
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
    """Finite field with integer encodings 0, ..., q-1."""

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

            self.add = [
                [
                    encode([
                        (x + y) % p
                        for x, y in zip(digits(a), digits(b))
                    ])
                    for b in range(q)
                ]
                for a in range(q)
            ]

            def multiply_poly(a, b):
                A, B = digits(a), digits(b)
                result = [0] * (2 * m - 1)

                for i, x in enumerate(A):
                    for j, y in enumerate(B):
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

        # Precompute additive inverses.
        self.neg = [
            next(b for b in range(q) if self.add[a][b] == 0)
            for a in range(q)
        ]

        # Precompute multiplicative inverses.
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
            f"Elements are encoded as integers 0,...,{self.q - 1}."
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
    print(f"Enter a list containing {k} rows.")
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
                        f"Every entry must be an integer from 0 to {q - 1}."
                    )

            print("\nAccepted generator matrix:")
            for row in G:
                print(" ", row)

            return G

        except (ValueError, SyntaxError, TypeError) as error:
            print("Invalid matrix:", error)
            print("Example: [[1, 0, 1], [0, 1, 1]]")


# ============================================================
# OPTIMIZATION: FIND AN INDEPENDENT ROW BASIS
# ============================================================

def row_basis(F, G, n):
    """
    Gaussian elimination over F_q.
    Returns independent rows spanning the same row space as G.
    """
    A = [row[:] for row in G]
    q = F.q
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

        # Normalize the pivot to 1.
        inverse = F.inv[A[rank][col]]

        A[rank] = [
            F.mul[inverse][x]
            for x in A[rank]
        ]

        # Eliminate entries below the pivot.
        pivot_row = A[rank]

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
# OPTIMIZATION: GENERATE CODEWORDS
# ============================================================

def span(F, basis, n):
    """
    Generate all codewords using an independent row basis.

    Avoids set construction and duplicate codewords.
    For rank r, generates exactly q^r codewords.
    """
    codewords = [tuple([0] * n)]

    for row in basis:
        old_words = codewords

        # Precompute every nonzero scalar multiple of this row.
        scaled_rows = []

        for c in range(1, F.q):
            scaled_rows.append(
                tuple(F.mul[c][x] for x in row)
            )

        new_words = []

        # Zero multiple is already represented by old_words.
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
# COMPLETE WEIGHT ENUMERATOR
# ============================================================

def cwe(F, code):
    """
    CWE coefficient key:
    (number of zeros, number of ones, ..., number of q-1s).
    """
    result = Counter()
    q = F.q

    for word in code:
        counts = [0] * q

        for symbol in word:
            counts[symbol] += 1

        result[tuple(counts)] += 1

    return result


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

def program_cwe():
    print("\n=== Complete Weight Enumerator (CWE) ===")

    F = ask_field()
    n = ask_int("Code length n: ", 1)
    number_of_codes = ask_int("Number of codes: ", 1)

    for idx in range(1, number_of_codes + 1):
        label = f"code {idx}"

        G = ask_generator_matrix(F, n, label)

        # Reduce the input matrix to an independent row basis.
        basis = row_basis(F, G, n)
        rank = len(basis)

        print(f"\nRank of generator matrix = {rank}")

        # Generate each codeword exactly once.
        code = span(F, basis, n)

        print(f"Dimension of code = {rank}")
        print(f"Number of codewords = {len(code)}")
        print(f"Expected number q^rank = {F.q ** rank}")

        variables = [f"x{a}" for a in range(F.q)]

        show_poly(
            cwe(F, code),
            variables,
            f"Complete Weight Enumerator of {label}"
        )

        if len(code) <= 64:
            print("\nCodewords:")
            for word in sorted(code):
                print(word)
        else:
            print(
                "\nCodewords not displayed because there are "
                "more than 64."
            )


if __name__ == "__main__":
    try:
        program_cwe()
    except (EOFError, KeyboardInterrupt):
        print("\nProgram terminated.")
        sys.exit(0)
