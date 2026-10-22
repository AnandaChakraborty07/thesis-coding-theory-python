
"""
Test of the Main Theorem (Theorem 4.2) for the average complete joint weight
enumerator.
"""

import sys
import ast
from fractions import Fraction
from itertools import product, permutations
from math import factorial, comb
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
# 3. ENUMERATORS
# ============================================================

def comp(u, q):
    c = [0] * q
    for x in u:
        c[x] += 1
    return tuple(c)


def bicomp(u, v, q):
    c = [0] * (q * q)
    for a, b in zip(u, v):
        c[a * q + b] += 1
    return tuple(c)


def cwe(F, code):
    """Complete weight enumerator coefficients A_r = #{u : comp(u) = r}."""
    return Counter(comp(u, F.q) for u in code)


def cjwe(F, c1, c2):
    """Complete joint weight enumerator {bicomposition: A_eta}."""
    q = F.q
    cnt = Counter()

    for u in c1:
        for v in c2:
            cnt[bicomp(u, v, q)] += 1

    return cnt


def multinom(parts):
    remaining = sum(parts)
    result = 1

    for x in parts:
        result *= comb(remaining, x)
        remaining -= x

    return result


# ============================================================
# 4. STEP 1: THE AVERAGE BY DEFINITION
# ============================================================

def matmul(F, A, B):
    n, m, l = len(A), len(B), len(B[0])
    C = [[0] * l for _ in range(n)]

    for i in range(n):
        for j in range(l):
            acc = 0
            for k in range(m):
                acc = F.add[acc][F.mul[A[i][k]][B[k][j]]]
            C[i][j] = acc

    return C


def monomial_matrices(F, n):
    """All (q-1)^n n! monomial matrices M = D P."""
    for sigma in permutations(range(n)):
        P = [[1 if j == sigma[i] else 0 for j in range(n)] for i in range(n)]

        for d in product(range(1, F.q), repeat=n):
            D = [[d[i] if i == j else 0 for j in range(n)] for i in range(n)]
            yield matmul(F, D, P)


def act(F, M, u):
    """u M := (M u^T)^T = (D (P u^T))^T."""
    n = len(u)
    out = []

    for i in range(n):
        acc = 0
        for j in range(n):
            acc = F.add[acc][F.mul[M[i][j]][u[j]]]
        out.append(acc)

    return tuple(out)


def average_by_definition(F, c1, c2, n):
    total = Counter()
    count = 0

    for M in monomial_matrices(F, n):
        count += 1
        c1M = [act(F, M, u) for u in c1]

        for key, val in cjwe(F, c1M, c2).items():
            total[key] += val

    assert count == (F.q - 1) ** n * factorial(n)

    return {k: Fraction(v, count) for k, v in total.items()}


# ============================================================
# 5. STEP 2: THE AVERAGE BY THEOREM 4.2
# ============================================================

def compositions(total, caps):
    if not caps:
        if total == 0:
            yield ()
        return

    for x in range(min(total, caps[0]) + 1):
        for rest in compositions(total - x, caps[1:]):
            yield (x,) + rest


def contingency_tables(r, s):
    """All q x q tables eta with row sums r and column sums s."""
    q = len(r)

    def rec(i, colrem):
        if i == q:
            if not any(colrem):
                yield []
            return

        for row in compositions(r[i], colrem):
            nxt = tuple(c - x for c, x in zip(colrem, row))
            for rest in rec(i + 1, nxt):
                yield [row] + rest

    yield from rec(0, tuple(s))


def average_by_theorem(F, c1, c2, n):
    """
    Right-hand side of Theorem 4.2. Terms with A^{C1 D}_r = 0 or
    A^{C2}_s = 0 vanish, so only compositions that occur are visited.
    """
    q = F.q
    A2 = cwe(F, c2)
    tables = {}
    result = {}

    for D in product(range(1, q), repeat=n):
        A1 = cwe(F, [
            tuple(F.mul[u[i]][D[i]] for i in range(n))
            for u in c1
        ])

        for r, A1r in A1.items():
            for s, A2s in A2.items():
                if (r, s) not in tables:
                    tables[(r, s)] = list(contingency_tables(r, s))

                for tab in tables[(r, s)]:
                    num = A1r * A2s

                    for i in range(q):
                        num *= multinom([tab[j][i] for j in range(q)])

                    coeff = Fraction(num, (q - 1) ** n * multinom(r))
                    key = tuple(x for row in tab for x in row)
                    result[key] = result.get(key, 0) + coeff

    return result


# ============================================================
# 6. STEP 3: COMPARISON
# ============================================================

def compare(p1, p2):
    keys = set(p1) | set(p2)

    return [
        (k, p1.get(k, 0), p2.get(k, 0))
        for k in keys
        if p1.get(k, 0) != p2.get(k, 0)
    ]


def run_test(F, c1, c2, n, verbose=True):
    q = F.q
    variables = make_variables(q)

    if verbose:
        print("\nStep 1: computing the average by definition ...")

    by_def = average_by_definition(F, c1, c2, n)

    if verbose:
        print("Step 2: computing the average by Theorem 4.2 ...")

    by_thm = average_by_theorem(F, c1, c2, n)

    if verbose:
        show_poly(by_def, variables, "Step 1: CJW^av by DEFINITION")
        show_poly(by_thm, variables, "Step 2: CJW^av by THEOREM 4.2")

    diffs = compare(by_def, by_thm)

    if not diffs:
        if verbose:
            print(f"\nStep 3: the two polynomials are IDENTICAL "
                  f"({len(by_def)} terms, all coefficients equal).")
        return True

    if verbose:
        print(f"\nStep 3: the polynomials DIFFER in {len(diffs)} "
              "coefficient(s):")

        for k, a, b in sorted(diffs, reverse=True)[:20]:
            print(f"   {mono(variables, k)}:  "
                  f"definition = {coefficient_text(a)},  "
                  f"theorem = {coefficient_text(b)}")

    return False


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
# 8. POLYNOMIAL DISPLAY  (subscripts and powers, e.g. x₀₀² + 2x₀₀x₁₁)
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


def coefficient_text(c):
    """Integer coefficient as is; a proper fraction as (a/b)."""
    c = Fraction(c)

    if c.denominator == 1:
        return str(c.numerator)

    return f"({c.numerator}/{c.denominator})"


def show_poly(table, variables, title):
    terms = []

    for key in sorted(table, reverse=True):
        c = table[key]

        if c == 0:
            continue

        monomial = mono(variables, key)
        text = coefficient_text(c)

        if monomial == "1":
            terms.append(text)
        elif text == "1":
            terms.append(monomial)
        else:
            terms.append(text + monomial)

    print(f"\n{title}")
    print("-" * len(title))
    print(" + ".join(terms) if terms else "0")
    print(f"\n#terms = {len(terms)}   "
          f"sum of coefficients = {coefficient_text(sum(table.values()))}")


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def banner():
    print("=" * 60)
    print("  Test of Theorem 4.2: definition versus theorem")
    print("=" * 60)
    print("You will be asked for:")
    print("  1. the field size q,")
    print("  2. the common code length n,")
    print("  3. a generator matrix for C1 and one for C2.")
    print("The codewords of C1 and C2 are printed, then the average CJWE is")
    print("computed by definition and by the theorem, and the two polynomials")
    print("are compared.")
    print("Type 'quit' at any prompt to leave.\n")


def print_codewords(F, code, label, limit=1000):
    print(f"\nCodewords of {label} ({len(code)} words):")

    for word in sorted(code)[:limit]:
        print(" ", word)

    if len(code) > limit:
        print(f"  ... {len(code) - limit} more codewords not shown.")


def size_warning(q, n):
    group = (q - 1) ** n * factorial(n)

    if group > 100_000:
        print(f"\nWarning: there are {group:,} monomial matrices, so Step 1 "
              "may take a long time.")


def program_test():
    banner()

    F = ask_field()
    q = F.q

    print("\nSTEP 2: the code length")
    n = ask_int("  Common code length n: ", 1)

    print("\nSTEP 3: the generator matrices")
    G1 = ask_generator_matrix(F, n, "C1")
    G2 = ask_generator_matrix(F, n, "C2")

    c1 = span(F, row_basis(F, G1, n), n)
    c2 = span(F, row_basis(F, G2, n), n)

    print_codewords(F, c1, "C1")
    print_codewords(F, c2, "C2")

    print(f"\n|C1| = {len(c1)},  |C2| = {len(c2)},  "
          f"monomial matrices = {(q - 1) ** n * factorial(n):,}")

    size_warning(q, n)

    ok = run_test(F, c1, c2, n, verbose=True)

    print("\nRESULT:",
          "Theorem verified for these codes." if ok
          else "Theorem FAILED for these codes.")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    try:
        program_test()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        sys.exit(0)
