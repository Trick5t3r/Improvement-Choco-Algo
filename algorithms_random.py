import numpy as np

def choco_gossip_over_time(size, nb_permutation, Q, p, num_iterations=None, eps_targ=0.01):
    """
    Implementation of the CHOCO-GOSSIP algorithm with graph animation.

    Parameters:
    - x_init: Initial values (array of shape n x d).
    - gamma: Stepsize (scalar).
    - W: Mixing matrix.
    - Q: Compression operator (function).
    - num_iterations: Number of iterations (optional).
    - eps_targ: Target accuracy (optional).

    Returns:
    - x_history: History of x values over iterations.
    """
    x_init = np.zeros((size, 1))
    for i in range(size // 2):
        x_init[i] = 1

    n, d = x_init.shape
    x = x_init.copy()
    x_hat = np.zeros_like(x)
    target = np.mean(x)
    x_history = [x.copy()]
    W_history = []
    rho_history = []
    t = -1

    while (num_iterations is None and np.abs(np.max(np.abs(x)) - target) > eps_targ) or (num_iterations and t < num_iterations - 1):
        W= generate_doubly_sto_matrix(size, nb_permutation)
        rho = compute_rho(W)
        beta = compute_beta(W)
        gamma = compute_gamma(rho, beta, p)
        
        
        t += 1
        x_hat_old = np.copy(x_hat)
        print(t, np.mean(x), np.abs(np.max(np.abs(x)) - target))
        for i in range(n):
            neighbors = np.where(W[i, :] != 0)[0]
            delta = sum(W[i, j] * (x_hat_old[j] - x_hat_old[i]) for j in neighbors)
            x[i] += gamma * delta
            for j in neighbors:
                x_hat[j] += Q(x[j] - x_hat_old[j])
        x_history.append(x.copy())
        W_history.append(W)
        rho_history.append(rho)

    return x_history, W_history, rho_history


def randomized_gossip(x, p):
    """
    Randomized Gossip Quantization function.

    Parameters:
    - x: Input vector (numpy array).
    - p: Probability of keeping the vector x.

    Returns:
    - Quantized vector Q(x).
    """
    return x if np.random.rand() < p else np.zeros_like(x)


def compute_rho(W):
    """
    Calculate spectral gap ρ := 1 − |λ2(W)|.

    Parameters:
    - W: Mixing matrix.

    Returns:
    - ρ: Spectral gap.
    """
    eigenvalues = np.linalg.eigvals(W)
    lambda_2 = sorted(np.abs(eigenvalues), reverse=True)[1]
    return 1 - lambda_2


def compute_beta(W):
    """
    Calculate spectral norm β := ‖I − W‖₂.

    Parameters:
    - W: Mixing matrix.

    Returns:
    - β: Spectral norm.
    """
    I_minus_W = np.eye(W.shape[0]) - W
    eigenvalues = np.linalg.eigvals(I_minus_W)
    return max(np.abs(eigenvalues))


def compute_gamma(rho, beta, delta):
    """
    Calculate γ := ρ²δ / (16ρ + ρ² + 4β² + 2ρβ² − 8ρδ).

    Parameters:
    - rho: Spectral gap.
    - beta: Spectral norm.
    - delta: Compression parameter.

    Returns:
    - γ: Computed value of γ.
    """
    numerator = rho**2 * delta
    denominator = 16 * rho + rho**2 + 4 * beta**2 + 2 * rho * beta**2 - 8 * rho * delta
    if denominator == 0:
        raise ValueError("Denominator is zero, cannot compute γ.")
    return numerator / denominator




def pas_choco_gossip_over_time(x_init, gamma, W, Q, num_iterations=None, eps_targ=0.01):
    """
    Implementation of the CHOCO-GOSSIP algorithm with graph animation.

    Parameters:
    - x_init: Initial values (array of shape n x d).
    - gamma: Stepsize (scalar).
    - W: Mixing matrix.
    - Q: Compression operator (function).
    - num_iterations: Number of iterations (optional).
    - eps_targ: Target accuracy (optional).

    Returns:
    - x_history: History of x values over iterations.
    """
    n, d = x_init.shape
    x = x_init.copy()
    x_hat = np.zeros_like(x)
    target = np.mean(x)
    x_history = [x.copy()]
    t = -1

    while (num_iterations is None and np.abs(np.max(np.abs(x)) - target) > eps_targ) or (num_iterations and t < num_iterations - 1):
        t += 1
        print(t, np.mean(x), np.abs(np.max(np.abs(x)) - target))
        for i in range(n):
            x_hat[i] = Q(x[i])
        for i in range(n):
            neighbors = np.where(W[i, :] != 0)[0]
            delta = sum(W[i, j] * (x_hat[j] - x[i]) for j in neighbors)
            x[i] += gamma * delta
        x_history.append(x.copy())
    return x_history


def generate_permutation_matrix(size):
    permutation = np.random.permutation(size)
    perm_matrix = np.eye(size)[permutation]
    return perm_matrix

def generate_doubly_sto_matrix(size,nb_permutation):
    
    # Création d'une nouvelle matrice de permutation de taille 6
    coefs=np.random.randint(1, size , size=nb_permutation)
    coefs=coefs/(coefs.sum())

    W=np.zeros((size,size))
    for i in range(nb_permutation):
        new_permutation_matrix = generate_permutation_matrix(size)
        W+=coefs[i]*new_permutation_matrix
    return W