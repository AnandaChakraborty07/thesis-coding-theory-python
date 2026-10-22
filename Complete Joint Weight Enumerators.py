
"""
Complete Joint Weight Enumerator (CJWE) of linear codes over F_q.
"""

import sys
import ast
from itertools import product
from collections import Counter


# ============================================================
# 1. FINITE FIELD F_q
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

    remainder = [x % p for x in a[:degree]]
    return remainder + [0] * (degree - len(remainder))


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
                [multiply_poly(a, b) for b in range(q)]
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
# 2. LINEAR ALGEBRA AND CODEWORDS
# ============================================================

def row_basis(F, G, n):
    """Return independent rows spanning the same code as G."""
    A = [list(row) for row in G]
    rank = 0

    for col in range(n):
        pivot = next(
            (i for i in range(rank, len(A)) if A[i][col] != 0),
            None
        )

        if pivot is None:
            continue

        A[rank], A[pivot] = A[pivot], A[rank]

        inverse = F.inv[A[rank][col]]
        A[rank] = [F.mul[inverse][x] for x in A[rank]]
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


def span(F, basis, n):
    """
    Generate each codeword exactly once from an independent basis.
    Number of codewords = q^rank.
    """
    codewords = [tuple([0] * n)]

    for row in basis:
        old_words = codewords

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
# 3. COMPLETE JOINT WEIGHT ENUMERATOR
# ============================================================

def cjwe(F, c1, c2):
    """
    Compute CJWE(C1, C2).

    The exponent of x[a,b] counts positions where u_i = a and v_i = b.
    """
    q = F.q
    cnt = Counter()

    for u in c1:
        for v in c2:
            counts = [0] * (q * q)

            for a, b in zip(u, v):
                counts[a * q + b] += 1

            cnt[tuple(counts)] += 1

    return cnt


# ============================================================
# 4. USER-FRIENDLY INPUT
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
    cells = [
        [F.element_name(x) if F.m > 1 else str(x) for x in row]
        for row in G
    ]
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
# 5. POLYNOMIAL DISPLAY  (subscripts and powers, e.g. x₀₀² + 2x₀₀x₁₁)
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
    """Monomial as a string, e.g. x₀₀²x₁₁.  Returns '1' if empty."""
    parts = []

    for variable, exponent in zip(variables, exponents):
        if exponent == 1:
            parts.append(variable)
        elif exponent > 1:
            parts.append(variable + str(exponent).translate(SUPERSCRIPT))

    return "".join(parts) if parts else "1"


def show_poly(table, variables, title):
    terms = []

    for exponents in sorted(table, reverse=True):
        coefficient = table[exponents]

        if coefficient == 0:
            continue

        monomial = mono(variables, exponents)

        if monomial == "1":
            terms.append(str(coefficient))
        elif coefficient == 1:
            terms.append(monomial)
        else:
            terms.append(f"{coefficient}{monomial}")

    print(f"\n{title}")
    print("-" * len(title))
    print(" + ".join(terms) if terms else "0")
    print(f"\n#terms = {len(terms)}")
    print(f"Sum of coefficients = {sum(table.values())}")


# ============================================================
# 6. MAIN PROGRAM
# ============================================================

def banner():
    print("=" * 60)
    print("  Complete Joint Weight Enumerator (CJWE)")
    print("=" * 60)
    print("You will be asked for:")
    print("  1. the field size q,")
    print("  2. the common code length n,")
    print("  3. the number of codes and a generator matrix for each.")
    print("The CJWE of every pair of codes is then printed.")
    print("Type 'quit' at any prompt to leave.\n")


def program_cjwe():
    banner()

    F = ask_field()
    q = F.q

    print("\nSTEP 2: the code length")
    n = ask_int("  Common code length n: ", 1)

    print("\nSTEP 3: the codes")
    num = ask_int("  Number of codes (at least 2): ", 2)

    codes = []

    for i in range(1, num + 1):
        G = ask_generator_matrix(F, n, f"C{i}")
        basis = row_basis(F, G, n)
        code = span(F, basis, n)
        codes.append(code)

        print(f"  C{i}: dimension {len(basis)}, {len(code)} codewords")

    variables = make_variables(q)

    for i in range(num):
        for j in range(i + 1, num):
            pairs = len(codes[i]) * len(codes[j])

            if pairs > 5_000_000:
                print(f"\nWarning: (C{i + 1}, C{j + 1}) needs {pairs:,} "
                      "word pairs. This may take a long time.")

            print(f"\nComputing CJWE of (C{i + 1}, C{j + 1}) ...")

            result = cjwe(F, codes[i], codes[j])

            show_poly(
                result,
                variables,
                f"CJWE of (C{i + 1}, C{j + 1})"
            )
            print("(x with subscript ab = number of positions where the "
                  "first word has a and the second has b)")

            actual_sum = sum(result.values())

            print(f"\nExpected coefficient sum = {pairs}")
            print(f"Actual coefficient sum   = {actual_sum}")
            print("Coefficient-sum check:",
                  "PASSED" if actual_sum == pairs else "FAILED")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    try:
        program_cjwe()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        sys.exit(0)
