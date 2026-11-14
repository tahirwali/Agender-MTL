"""
QUBO attribute selection (minimum redundancy, maximum relevance):

    E(x) = -alpha * sum_i R_i x_i  +  beta * sum_{i<j} Q_ij x_i x_j  +  lambda * (sum_i x_i - k)^2

R_i is the normalised mutual information (NMI) between attribute i and the label, Q_ij the NMI between
attributes i and j. The energy is minimised by simulated annealing on a CPU (6 restarts, 2e4 single-bit
flips each, geometric cooling from T = 5 to T = 0.01).

The matrix, the energy and the annealer are kept separate so each can be tested on its own.
"""
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def discretise(column, bins=10):
    """Quantile bins for a numeric column (ties merged); columns with few values are used as they are."""
    column = np.asarray(column, dtype=float)
    if len(np.unique(column)) <= bins:
        return np.unique(column, return_inverse=True)[1]
    edges = np.unique(np.quantile(column, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(edges, column, side="right")


def nmi_terms(X, label, bins=10):
    """R (relevance of each column to the label) and Q (NMI between every pair of columns)."""
    X = np.asarray(X, dtype=float)
    codes = [discretise(X[:, i], bins) for i in range(X.shape[1])]
    relevance = np.array([normalized_mutual_info_score(label, c) for c in codes])
    d = len(codes)
    redundancy = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            redundancy[i, j] = redundancy[j, i] = normalized_mutual_info_score(codes[i], codes[j])
    return relevance, redundancy


def qubo_matrix(relevance, redundancy, k, alpha=1.0, beta=0.05, lam=1.5):
    """Upper-triangular matrix M and constant c with E(x) = x^T M x + c for binary x.

    Expanding lambda (sum x - k)^2 with x_i^2 = x_i gives lambda (1 - 2k) on the diagonal and
    2 lambda on each pair, plus the constant lambda k^2."""
    d = len(relevance)
    matrix = np.triu(beta * redundancy + 2 * lam * np.ones((d, d)), k=1)
    matrix[np.diag_indices(d)] = -alpha * np.asarray(relevance) + lam * (1 - 2 * k)
    return matrix, lam * k * k


def energy(x, matrix, constant=0.0):
    x = np.asarray(x, dtype=float)
    return float(x @ matrix @ x + constant)


def anneal(matrix, constant=0.0, restarts=6, iterations=20_000, t_start=5.0, t_end=0.01, seed=0, swaps=False):
    """Simulated annealing. Returns the best x found and its energy.

    swaps=False is the plain annealer: every move flips one bit. With a cardinality
    penalty, changing which attributes are chosen (at a fixed number) then means passing through a
    state with one attribute more or less, which costs about lambda; once the temperature is low such
    moves stop and the search freezes. swaps=True makes every second move a swap (one chosen attribute
    out, one other in), which keeps the number of attributes and only pays the real difference."""
    rng = np.random.default_rng(seed)
    d = matrix.shape[0]
    full = matrix + matrix.T - np.diag(np.diag(matrix))     # symmetric pair weights, diagonal once
    diagonal = np.diag(matrix)
    temperatures = t_start * (t_end / t_start) ** (np.arange(iterations) / max(iterations - 1, 1))

    def flip_change(x, i):
        # change in energy if bit i flips: (1 - 2 x_i) * (M_ii + sum_{j != i} M_ij x_j)
        return (1 - 2 * x[i]) * (diagonal[i] + full[i] @ x - full[i, i] * x[i])

    best_x, best_e = None, np.inf
    for _ in range(restarts):
        x = rng.integers(0, 2, d).astype(float)
        e = energy(x, matrix, constant)
        if e < best_e:
            best_x, best_e = x.copy(), e
        flips = rng.integers(0, d, iterations)
        draws = rng.random(iterations)
        for step, (t, i, u) in enumerate(zip(temperatures, flips, draws)):
            if swaps and step % 2 == 1 and 0 < x.sum() < d:
                i = rng.choice(np.flatnonzero(x == 1))
                j = rng.choice(np.flatnonzero(x == 0))
                delta = flip_change(x, i)
                x[i] = 0
                delta += flip_change(x, j)
                x[i] = 1
                if delta <= 0 or u < np.exp(-delta / t):
                    x[i], x[j] = 0, 1
                    e += delta
                    if e < best_e:
                        best_x, best_e = x.copy(), e
                continue
            delta = flip_change(x, i)
            if delta <= 0 or u < np.exp(-delta / t):
                x[i] = 1 - x[i]
                e += delta
                if e < best_e:
                    best_x, best_e = x.copy(), e
    return best_x.astype(int), best_e


def select(X, label, k, alpha=1.0, beta=0.05, lam=1.5, seed=0, **annealing):
    """Indices of the attributes chosen by the QUBO, plus R and Q for reporting."""
    relevance, redundancy = nmi_terms(X, label)
    matrix, constant = qubo_matrix(relevance, redundancy, k, alpha, beta, lam)
    x, e = anneal(matrix, constant, seed=seed, **annealing)
    return np.flatnonzero(x), {"relevance": relevance, "redundancy": redundancy, "energy": e}
