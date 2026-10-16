
#!/usr/bin/env python3
"""
Average Complete Joint Weight Enumerator over the monomial group.

Computes:
    CJWE^av_(C1,C2) = 1/((q-1)^n n!) *
        sum_M CJWE_(C1 M,C2)

Optimizations:
1. No explicit monomial matrices.
2. No matrix multiplication for group actions.
3. Gaussian elimination removes dependent generator rows.
4. Efficient codeword generation.
5. Diagonal scalings are aggregated by symbol compositions.
6. Cached multinomial coefficients.
7. Lazy contingency-table generation.
8. Exact rational coefficients using Fraction.
9. Generator matrices accepted as list of lists.
"""

import sys
import ast
from itertools import product
from math import comb, factorial
from fractions import Fraction
from collections import Counter
from functools import lru_cache


# ============================================================
# 1. FINITE FIELD F_q
# ============================================================

def prime_power(q):
    for p in range(2, q + 1):
        if q % p == 0:
            m, t = 0, q
            while t % p == 0:
                t //= p
                m += 1
            return (p, m) if t == 1 else None
    return None


def poly_mod(a, f, p):
    """Polynomial remainder; coefficients are in low-degree-first order."""
    a = a[:]
    degree = len(f) - 1

    for i in range(len(a) - 1, degree - 1, -1):
        c = a[i] % p
        if c:
            for j in range(degree + 1):
                a[i - degree + j] = (
                    a[i - degree + j] - c * f[j]
                ) % p

    remainder = [x % p for x in a[:degree]]
    return remainder + [0] * (degree - len(remainder))


def is_irreducible(f, p):
    """Test irreducibility by checking possible monic divisors."""
    degree = len(f) - 1

    for d in range(1, degree // 2 + 1):
        for low in product(range(p), repeat=d):
            divisor = list(low) + [1]
            if not any(poly_mod(f, divisor, p)):
                return False

    return True


class GF:
    def __init__(self, q):
        pm = prime_power(q)
        if pm is None:
            raise ValueError(f"{q} is not a prime power")

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
                raise ValueError("Could not find an irreducible polynomial.")

            f = self.modpoly

            def digits(x):
                return [(x // (p ** i)) % p for i in range(m)]

            def encode(coeffs):
                return sum(c * (p ** i) for i, c in enumerate(coeffs))

            digit_table = [digits(x) for x in range(q)]

            self.add = [
                [
                    encode([
                        (x + y) % p
                        for x, y in zip(digit_table[a], digit_table[b])
                    ])
                    for b in range(q)
                ]
                for a in range(q)
            ]

            def multiply_polynomials(a, b):
                A, B = digit_table[a], digit_table[b]
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
                [multiply_polynomials(a, b) for b in range(q)]
                for a in range(q)
            ]

        self.neg = [
            next(b for b in range(q) if self.add[a][b] == 0)
            for a in range(q)
        ]

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
            f"F_{self.q} = F_{self.p}[t]/({terms}); "
            "elements are encoded by base-p coefficients."
        )


# ============================================================
# 2. CODE GENERATION
# ============================================================

def row_basis(F, G, n):
    """Return a linearly independent row basis using Gaussian elimination."""
    if not G:
        return []

    A = [list(row) for row in G]
    pivot_row = 0

    for col in range(n):
        pivot = next(
            (i for i in range(pivot_row, len(A)) if A[i][col] != 0),
            None
        )

        if pivot is None:
            continue

        A[pivot_row], A[pivot] = A[pivot], A[pivot_row]

        inv_pivot = F.inv[A[pivot_row][col]]
        A[pivot_row] = [
            F.mul[inv_pivot][x] for x in A[pivot_row]
        ]

        for i in range(len(A)):
            if i == pivot_row:
                continue

            c = A[i][col]
            if c == 0:
                continue

            A[i] = [
                F.add[A[i][j]][F.neg[F.mul[c][A[pivot_row][j]]]]
                for j in range(n)
            ]

        pivot_row += 1
        if pivot_row == len(A):
            break

    return A[:pivot_row]


def span(F, G, n):
    """Generate all codewords from the independent row basis of G."""
    basis = row_basis(F, G, n)
    q = F.q
    words = {(0,) * n}

    # Precompute all scalar multiples of every basis row.
    scaled_rows = [
        [
            tuple(F.mul[c][x] for x in row)
            for c in range(q)
        ]
        for row in basis
    ]

    for scaled in scaled_rows:
        new_words = set()

        for word in words:
            for row in scaled:
                new_words.add(tuple(
                    F.add[word[j]][row[j]]
                    for j in range(n)
                ))

        words = new_words

    return sorted(words)


# ============================================================
# 3. COMPLETE WEIGHT ENUMERATOR
# ============================================================

def comp(word, q):
    counts = [0] * q
    for a in word:
        counts[a] += 1
    return tuple(counts)


def cwe(F, code):
    return Counter(comp(word, F.q) for word in code)


# ============================================================
# 4. MULTINOMIAL COEFFICIENTS AND COMPOSITIONS
# ============================================================

@lru_cache(maxsize=None)
def multinom(parts):
    """Compute sum(parts)! / product(parts[i]!) exactly."""
    remaining = sum(parts)
    result = 1

    for x in parts:
        result *= comb(remaining, x)
        remaining -= x

    return result


def weak_compositions(total, parts):
    """All nonnegative tuples of length parts summing to total."""
    if parts == 0:
        if total == 0:
            yield ()
        return

    if parts == 1:
        yield (total,)
        return

    for x in range(total + 1):
        for rest in weak_compositions(total - x, parts - 1):
            yield (x,) + rest


def bounded_compositions(total, caps):
    """Tuples x satisfying sum(x)=total and 0 <= x[i] <= caps[i]."""
    if not caps:
        if total == 0:
            yield ()
        return

    for x in range(min(total, caps[0]) + 1):
        for rest in bounded_compositions(total - x, caps[1:]):
            yield (x,) + rest


# ============================================================
# 5. CONTINGENCY TABLES
# ============================================================

def contingency_tables(r, s):
    """
    Generate q x q tables eta with row sums r and column sums s.
    The final row is determined by the remaining column sums.
    """
    q = len(r)

    if sum(r) != sum(s):
        return

    def rec(i, colrem, rows):
        if i == q - 1:
            if sum(colrem) == r[i]:
                yield tuple(rows + [tuple(colrem)])
            return

        for row in bounded_compositions(r[i], colrem):
            nxt = tuple(
                colrem[j] - row[j]
                for j in range(q)
            )
            yield from rec(i + 1, nxt, rows + [row])

    yield from rec(0, tuple(s), [])


# ============================================================
# 6. OPTIMIZED AVERAGE CJWE
# ============================================================

def average_cjwe(F, c1, c2, n):
    """
    Exact average CJWE over the entire monomial group.

    Equivalent to explicitly averaging CJWE_(C1 M,C2) over all M,
    but does not construct or enumerate monomial matrices.
    """
    q = F.q

    # Count C1 words according to their complete symbol composition.
    c1_cwe = cwe(F, c1)

    # Aggregate by the number of zero coordinates.
    zero_distribution = Counter()
    for composition, multiplicity in c1_cwe.items():
        zero_distribution[composition[0]] += multiplicity

    # S[r] = sum_D A_r^(C1 D), computed by composition counting.
    S = Counter()

    for zeros, number_of_words in zero_distribution.items():
        nonzeros = n - zeros
        zero_factor = (q - 1) ** zeros

        for tail in weak_compositions(nonzeros, q - 1):
            r = (zeros,) + tail
            S[r] += (
                number_of_words
                * zero_factor
                * multinom(tail)
            )

    # Complete weight enumerator of the fixed code C2.
    A2 = cwe(F, c2)

    diagonal_count = (q - 1) ** n
    result = {}

    for r, Sr in S.items():
        denominator = diagonal_count * multinom(r)

        for s, As in A2.items():
            for table in contingency_tables(r, s):

                # Product of multinomial coefficients of the columns.
                table_factor = 1

                for col in range(q):
                    column = tuple(
                        table[row][col] for row in range(q)
                    )
                    table_factor *= multinom(column)

                key = tuple(
                    table[i][j]
                    for i in range(q)
                    for j in range(q)
                )

                numerator = Sr * As * table_factor

                result[key] = (
                    result.get(key, Fraction(0, 1))
                    + Fraction(numerator, denominator)
                )

    return {key: value for key, value in result.items() if value}


# ============================================================
# 7. INPUT HELPERS
# ============================================================

def ask_int(prompt, lo=None, hi=None):
    while True:
        try:
            value = int(input(prompt).strip())

            if (lo is None or value >= lo) and (
                hi is None or value <= hi
            ):
                return value

        except ValueError:
            pass

        print("  Invalid input, try again.")


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

    while True:
        k = ask_int(
            f"[{label}] Number of generator-matrix rows: ",
            1
        )

        print(f"[{label}] Enter the full matrix as a list of lists.")
        print("Example: [[1, 0, 1], [0, 1, 1]]")
        print(f"Required dimensions: {k} x {n}; entries 0,...,{q-1}.")

        try:
            G = ast.literal_eval(input("  G = ").strip())

            valid = (
                isinstance(G, list)
                and len(G) == k
                and all(
                    isinstance(row, list)
                    and len(row) == n
                    and all(
                        type(x) is int and 0 <= x < q
                        for x in row
                    )
                    for row in G
                )
            )

            if valid:
                return G

        except (ValueError, SyntaxError):
            pass

        print("  Invalid matrix. Please try again.")


# ============================================================
# 8. POLYNOMIAL DISPLAY
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

    for key in sorted(table, reverse=True):
        coefficient = table[key]

        if coefficient == 0:
            continue

        if isinstance(coefficient, Fraction):
            if coefficient.denominator == 1:
                coeff_text = str(coefficient.numerator)
            else:
                coeff_text = (
                    f"({coefficient.numerator}/"
                    f"{coefficient.denominator})"
                )
        else:
            coeff_text = str(coefficient)

        monomial = mono(variables, key)

        if monomial == "1":
            terms.append(coeff_text)
        else:
            terms.append(f"{coeff_text}*{monomial}")

    print(" + ".join(terms) if terms else "0")
    print(f"\n#terms = {len(terms)}")
    print(f"Sum of coefficients = {sum(table.values())}")


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def program_avg():
    print("\n=== Average CJWE over the monomial group ===")

    F = ask_field()
    n = ask_int("Common code length n: ", 1)

    G1 = ask_generator_matrix(F, n, "C1")
    G2 = ask_generator_matrix(F, n, "C2")

    B1 = row_basis(F, G1, n)
    B2 = row_basis(F, G2, n)

    c1 = span(F, B1, n)
    c2 = span(F, B2, n)

    q = F.q
    group_size = (q - 1) ** n * factorial(n)

    print(f"\nDimension of C1 = {len(B1)}")
    print(f"|C1| = {len(c1)}")
    print(f"Dimension of C2 = {len(B2)}")
    print(f"|C2| = {len(c2)}")
    print(f"Number of monomial matrices = {group_size:,}")

    variables = [
        f"x[{a},{b}]"
        for a in range(q)
        for b in range(q)
    ]

    avg = average_cjwe(F, c1, c2, n)

    show_poly(
        avg,
        variables,
        "Average CJWE: C1 transformed, C2 fixed"
    )

    expected_sum = len(c1) * len(c2)
    actual_sum = sum(avg.values())

    print(f"\nExpected coefficient sum = {expected_sum}")
    print(f"Actual coefficient sum   = {actual_sum}")
    print("Coefficient-sum check:", "PASSED" if actual_sum == expected_sum else "FAILED")


if __name__ == "__main__":
    try:
        program_avg()
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)
