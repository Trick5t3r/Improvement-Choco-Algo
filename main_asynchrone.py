import numpy as np
from algorithms_asynchrone import choco_gossip_over_time_asynchrone, randomized_gossip, compute_rho, compute_beta, compute_gamma
from visualization_dirige_asynchronous import animate_gossip

# Some gossip matrix generators
def genere_etoile_dirigee():
    n, d = 6, 1
    x_init = np.zeros((n, d))
    for i in range(n // 2):
        x_init[i] = 1

    W = np.zeros((n, n))

    for i in range(n):
        W[i, i] = 0.4  
        W[i, (i - 1) % n] = 0.3 
        W[i, (i + 1) % n] = 0.2 
        W[i, (i + 2) % n] = 0.1  

    W = W / W.sum(axis=1, keepdims=True)

    column_sums = W.sum(axis=0)
    for j in range(n):
        W[:, j] /= column_sums[j]
    return W, x_init

def genre_sans_retour_trois():
    n, d = 3, 1
    x_init = np.zeros((n, d))
    for i in range(n // 2):
        x_init[i] = 1

    W= np.array([[0.3, 0.3, 0.4], 
                 [0, 0.4 , 0.6],
                 [0.7, 0.3, 0]])
    return W, x_init

def genere_matrice_gossip_cyclique(n):
    """
    Génère une matrice double stochastique symétrique pour une topologie cyclique.

    Arguments :
        n : int, nombre de nœuds dans le cycle.

    Retourne :
        W : np.ndarray, matrice double stochastique symétrique de taille (n, n).
    """
    W = np.zeros((n, n))
    
    poids_principal = 0.5  
    poids_voisin_direct = 0.25  
    poids_voisin_lointain = 0.0  

    if poids_principal + 2 * poids_voisin_direct + 2 * poids_voisin_lointain != 1.0:
        raise ValueError("La somme des poids doit être égale à 1.0 pour garantir la stochasticité.")

    for i in range(n):
        W[i, i] = poids_principal 
        W[i, (i - 1) % n] = poids_voisin_direct 
        W[i, (i + 1) % n] = poids_voisin_direct 
        W[i, (i - 2) % n] = poids_voisin_lointain  
        W[i, (i + 2) % n] = poids_voisin_lointain 

    for i in range(n):
        W[i, :] /= W[i, :].sum()
    for j in range(n):
        W[:, j] /= W[:, j].sum()

    if not np.allclose(W, W.T, atol=1e-10):
        raise ValueError("La matrice générée n'est pas symétrique.")
    
    x_init = np.zeros((n, 1))
    for i in range(n // 2):
        x_init[i] = 1

    return W, x_init

def genere_matrice_gossip_grille(n):
    """
    Génère une matrice double stochastique symétrique pour une topologie en grille carrée.

    Arguments :
        n : int, taille de la grille (nombre de nœuds par côté, total n*n).

    Retourne :
        W : np.ndarray, matrice double stochastique symétrique de taille (n*n, n*n).
        x_init : np.ndarray, vecteur initial des états des nœuds.
    """
    total_nodes = n * n 
    W = np.zeros((total_nodes, total_nodes))

    def index(row, col):
        return row * n + col

    for row in range(n):
        for col in range(n):
            current_node = index(row, col)
            
            W[current_node, current_node] = 0.4
            
            if col > 0:
                left_node = index(row, col - 1)
                W[current_node, left_node] = 0.15
                W[left_node, current_node] = 0.15 
            
            if col < n - 1:
                right_node = index(row, col + 1)
                W[current_node, right_node] = 0.15
                W[right_node, current_node] = 0.15 
            
            if row > 0:
                top_node = index(row - 1, col)
                W[current_node, top_node] = 0.15
                W[top_node, current_node] = 0.15 
            
            if row < n - 1:
                bottom_node = index(row + 1, col)
                W[current_node, bottom_node] = 0.15
                W[bottom_node, current_node] = 0.15  

    if not np.allclose(W, W.T, atol=1e-10):
        raise ValueError("La matrice générée n'est pas symétrique.")

    # Initialisation de x_init avec une moitié de nœuds activés
    x_init = np.zeros((total_nodes, 1))
    for i in range(total_nodes // 2):
        x_init[i] = 1

    return W, x_init



p = 0.5
Q = lambda x: randomized_gossip(x, p)

size = 10
nb_permutation = 2

# Run Algorithm
x_history, W_history, rho_history = choco_gossip_over_time_asynchrone(size, nb_permutation, Q, p)

# Animate
animate_gossip(W_history, x_history)

"""import matplotlib.pyplot as plt

plt.hist(rho_history, bins=50, alpha=0.5, color='blue')
plt.title("Histogramme des tirages")
plt.xlabel("Valeur")
plt.ylabel("Fréquence")
plt.show()"""

x_history=np.array(x_history)
x_history_log=np.log(x_history[:,0]-0.5)

# Animate
import matplotlib.pyplot as plt

plt.plot(x_history_log)
plt.show()