import numpy as np
import time

def choco_gossip_over_time_asynchrone(size, nb_permutation, Q, p, num_iterations=None, eps_targ=0.01):
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
    next_tick_times, current_time = None, None

    while (num_iterations is None and np.abs(np.max(np.abs(x)) - target) > eps_targ) or (num_iterations and t < num_iterations - 1):
        t += 1

        W, next_tick_times, current_time = generate_doubly_sto_matrix_asynchrone(size, next_tick_times, current_time)
        rho = 0
        beta = 0
        gamma = 1/2
        
        
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

        if t > 300:
            break
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



def generate_doubly_sto_matrix_asynchrone(n_agents, next_tick_times=None, current_time=0, rate=1):
    """
    Simule un système où les agents interagissent via des horloges Poissoniennes et génère une matrice de gossip.

    Parameters:
        n_agents (int): Nombre d'agents.
        rate (float): Taux de ticking pour chaque agent (Poisson rate).
        max_steps (int): Nombre maximal d'étapes de simulation.

    Returns:
        np.array: Matrice de gossip basée sur les interactions.
    """
    # Initialisation des horloges locales pour chaque agent
    if next_tick_times is None:
        next_tick_times = np.random.exponential(1 / rate, n_agents)
        current_time = 0
    adjacency_matrix = np.eye(n_agents)

    i_agent = np.argmin(next_tick_times)  # Vérifie si l'agent i a un tick
    next_tick_times[i_agent] += np.random.exponential(1 / rate)  # Planifie le prochain tick
    # Sélectionne un voisin au hasard
    j = np.random.choice([k for k in range(n_agents) if k != i_agent])
    adjacency_matrix[i_agent, i_agent] = 0
    adjacency_matrix[j, j] = 0
    adjacency_matrix[i_agent, j] = 1
    adjacency_matrix[j, i_agent] = 1

    return adjacency_matrix, next_tick_times, current_time
