#!/usr/bin/env python3
"""
Average Complete Joint Weight Enumerator over the monomial group.

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
            f"F_{self.q} = F_{self.p}[t]/({terms}), "
            "where t is a root of this polynomial"
        )

    def element_name(self, x):
        """Readable name of element x, e.g. 0, 1, t, t+1, 2t+1."""
        if self.m == 1:
            return str(x)

        digits = [(x // (self.p ** i)) % self.p for i in range(self.m)]
        parts = []

        for i in range(self.m - 1, -1, -1):
            c = digits[i]
            if c == 0:
                continue
            if i == 0:
                parts.append(str(c))
            elif i == 1:
                parts.append("t" if c == 1 else f"{c}t")
            else:
                parts.append(f"t^{i}" if c == 1 else f"{c}t^{i}")

        return "+".join(parts) if parts else "0"


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
# 7. USER-FRIENDLY INPUT
# ============================================================

QUIT_WORDS = {"quit", "exit"}


def read_line(prompt):
    """input() that lets the user type 'quit' or 'exit' at any prompt."""
    text = input(prompt).strip()
    if text.lower() in QUIT_WORDS:
        raise EOFError
    return text


def ask_int(prompt, lo=None, hi=None):
    while True:
        text = read_line(prompt)

        try:
            value = int(text)
        except ValueError:
            print("  Please enter a whole number.")
            continue

        if lo is not None and value < lo:
            print(f"  The number must be at least {lo}.")
        elif hi is not None and value > hi:
            print(f"  The number must be at most {hi}.")
        else:
            return value


def ask_field():
    print("STEP 1: the finite field F_q")
    print("  q must be a prime power: 2, 3, 4, 5, 7, 8, 9, 11, 13, 16, ...")

    while True:
        q = ask_int("  Field size q: ", 2)

        try:
            F = GF(q)
        except ValueError:
            print(f"  {q} is not a prime power. Try again.")
            continue

        print(" ", F.describe())

        if F.m > 1:
            print("  Enter field elements as the numbers 0,...,"
                  f"{q - 1}. They stand for:")
            for x in range(q):
                print(f"    {x:>3}  <->  {F.element_name(x)}")

        return F


def parse_row(text, n, q):
    """
    Parse one matrix row. Accepted styles:
        1 0 1        1,0,1        [1, 0, 1]        101  (only when q <= 10)
    """
    cleaned = text.strip().strip("[]()").strip()

    if "," in cleaned or " " in cleaned:
        tokens = cleaned.replace(",", " ").split()
    elif q <= 10 and len(cleaned) == n:
        tokens = list(cleaned)
    else:
        tokens = cleaned.split()

    try:
        row = [int(t) for t in tokens]
    except ValueError:
        raise ValueError("a row may only contain whole numbers.")

    if len(row) != n:
        raise ValueError(f"expected {n} entries but got {len(row)}.")

    bad = [x for x in row if not 0 <= x < q]
    if bad:
        raise ValueError(
            f"entries must be between 0 and {q - 1}; found {bad[0]}."
        )

    return row


def parse_whole_matrix(text, k, n, q):
    """Parse a pasted list of lists such as [[1,0,1],[0,1,1]]."""
    try:
        G = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        raise ValueError("could not read that as a list of lists.")

    if not isinstance(G, (list, tuple)) or len(G) != k:
        raise ValueError(f"expected {k} rows.")

    return [parse_row(str(list(row)), n, q) for row in G]


def print_matrix(F, G):
    cells = [[F.element_name(x) if F.m > 1 else str(x) for x in row]
             for row in G]
    width = max((len(c) for row in cells for c in row), default=1)

    for row in cells:
        print("    [ " + "  ".join(c.rjust(width) for c in row) + " ]")


def ask_generator_matrix(F, n, label):
    q = F.q

    print(f"\n  Generator matrix of {label}")

    while True:
        k = ask_int(f"  Number of rows of G for {label}: ", 1)

        print(f"  Enter the {k} rows one at a time, each with {n} entries "
              f"from 0 to {q - 1}.")
        print("  Examples:  1 0 1     or     1,0,1     "
              + ("or     101" if q <= 10 else ""))
        print("  (Or paste the whole matrix at once, e.g. [[1,0,1],[0,1,1]])")

        G = []
        i = 0

        while i < k:
            text = read_line(f"    Row {i + 1} of {k}: ")

            try:
                if i == 0 and text.startswith("[["):
                    G = parse_whole_matrix(text, k, n, q)
                    break

                G.append(parse_row(text, n, q))
                i += 1

            except ValueError as error:
                print(f"    Problem: {error} Please re-enter this row.")

        rank = len(row_basis(F, G, n))

        if rank == 0:
            print("  Invalid matrix: every row is zero, so the code has "
                  "no nonzero words. Please enter the matrix again.\n")
            continue

        print(f"\n  {label} generator matrix ({k} x {n}) accepted:")
        print_matrix(F, G)

        if rank < k:
            print(f"  Note: the rows are linearly dependent; "
                  f"the code has dimension {rank}, not {k}. "
                  "Dependent rows are removed automatically.")

        return G


# ============================================================
# 8. POLYNOMIAL DISPLAY  (subscripts and powers, e.g. x₀₀² + 2x₀₀x₁₀)
# ============================================================

SUBSCRIPT = str.maketrans("0123456789,", "₀₁₂₃₄₅₆₇₈₉,")
SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def make_variables(q):
    """
    Variable names for the q*q symbol pairs (a,b).
    For q <= 10 the index is two digits (x₀₀, x₁₂, ...); for q > 10 the two
    numbers are separated by a comma and bracketed: x₍₁₀,₁₁₎.
    """
    sep = "" if q <= 10 else ","
    names = []

    for a in range(q):
        for b in range(q):
            sub = f"{a}{sep}{b}".translate(SUBSCRIPT)
            if q > 10:
                sub = "₍" + sub + "₎"
            names.append("x" + sub)

    return names


def mono(variables, exponents):
    """Monomial as a string, e.g. x₀₀²x₁₀.  Returns '1' if empty."""
    parts = []

    for variable, exponent in zip(variables, exponents):
        if exponent == 1:
            parts.append(variable)
        elif exponent > 1:
            parts.append(variable + str(exponent).translate(SUPERSCRIPT))

    return "".join(parts) if parts else "1"


def coefficient_text(coefficient):
    """Integer coefficient as is; a proper fraction as (a/b)."""
    if isinstance(coefficient, Fraction):
        if coefficient.denominator == 1:
            return str(coefficient.numerator)
        return f"({coefficient.numerator}/{coefficient.denominator})"
    return str(coefficient)


def show_poly(table, variables, title):
    terms = []

    for key in sorted(table, reverse=True):
        coefficient = table[key]

        if coefficient == 0:
            continue

        monomial = mono(variables, key)
        coeff = coefficient_text(coefficient)

        if monomial == "1":
            terms.append(coeff)
        elif coeff == "1":
            terms.append(monomial)
        else:
            terms.append(coeff + monomial)

    print(f"\n{title}")
    print("-" * len(title))
    print(" + ".join(terms) if terms else "0")
    print(f"\n#terms = {len(terms)}")
    print(f"Sum of coefficients = {sum(table.values())}")


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def banner():
    print("=" * 60)
    print("  Average Complete Joint Weight Enumerator (monomial group)")
    print("=" * 60)
    print("You will be asked for:")
    print("  1. the field size q,")
    print("  2. the common code length n,")
    print("  3. a generator matrix for C1 and one for C2.")
    print("Type 'quit' at any prompt to leave.\n")


def program_avg():
    banner()

    F = ask_field()
    q = F.q

    print("\nSTEP 2: the code length")
    n = ask_int("  Common code length n: ", 1)

    print("\nSTEP 3: the generator matrices")
    G1 = ask_generator_matrix(F, n, "C1")
    G2 = ask_generator_matrix(F, n, "C2")

    B1 = row_basis(F, G1, n)
    B2 = row_basis(F, G2, n)

    if q ** len(B1) > 2_000_000 or q ** len(B2) > 2_000_000:
        print(f"\nWarning: |C1| = {q ** len(B1):,} and |C2| = "
              f"{q ** len(B2):,}. This may take a long time.")

    print("\nComputing ...")

    c1 = span(F, B1, n)
    c2 = span(F, B2, n)
    group_size = (q - 1) ** n * factorial(n)

    print(f"\nDimension of C1 = {len(B1)}   (|C1| = {len(c1)})")
    print(f"Dimension of C2 = {len(B2)}   (|C2| = {len(c2)})")
    print(f"Number of monomial matrices = {group_size:,}")

    avg = average_cjwe(F, c1, c2, n)

    show_poly(
        avg,
        make_variables(q),
        "Average CJWE: C1 transformed, C2 fixed"
    )

    expected_sum = len(c1) * len(c2)
    actual_sum = sum(avg.values())

    print(f"\nExpected coefficient sum = {expected_sum}")
    print(f"Actual coefficient sum   = {actual_sum}")
    print("Coefficient-sum check:",
          "PASSED" if actual_sum == expected_sum else "FAILED")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    try:
        program_avg()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        sys.exit(0)
