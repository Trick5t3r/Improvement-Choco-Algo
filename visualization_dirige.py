import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def animate_gossip(W, x_history):
    """
    Create an animation of the gossip process.

    Parameters:
    - W: Mixing matrix defining the graph structure.
    - x_history: List of x values at each iteration.
    """
    G = nx.DiGraph()
    n = W.shape[0]
    
    # Add edges based on W
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                G.add_edge(i, j, weight=round(W[i, j], 2))
            if W[j, i] > 0:
                G.add_edge(j, i, weight=round(W[j, i], 2))

    # Create positions for the nodes
    pos = nx.spring_layout(G, seed=42)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    node_labels = {}
    node_colors = []

    def update(frame):
        ax.clear()
        current_values = x_history[frame]
    
        # Define custom RGB colors for each node
        node_colors = [
            (1 - np.clip(current_values[i][0], 0, 1), 0, np.clip(current_values[i][0], 0, 1))  # Blue to Red gradient
            for i in range(n)
        ]
        
        node_labels = {i: f"{current_values[i][0]:.2f}" for i in range(n)}

        # Calculate the average value of the nodes
        avg_value = np.mean([current_values[i][0] for i in range(n)])

        # Dessiner les nœuds
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=800)
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_color="white")
        
        # Dessiner les arêtes sous forme d'arcs avec les poids
        for (u, v, d) in G.edges(data=True):
            weight = d['weight']
            arrowstyle = f"-|>,head_width=0.3,head_length=0.5"
            color = "black"
            
            # Tracer l'arête
            ax.annotate(
                "", 
                xy=pos[v], xycoords="data",
                xytext=pos[u], textcoords="data",
                arrowprops=dict(
                    arrowstyle=arrowstyle, color=color, lw=1.5,
                    shrinkA=15, shrinkB=15, connectionstyle="arc3,rad=0.2"
                )
            )
            
            # Ajouter le poids sur l'arête
            mid_point = (
                (pos[u][0] + pos[v][0]) / 2,  # Coordonnée x du milieu
                (pos[u][1] + pos[v][1]) / 2   # Coordonnée y du milieu
            )
            text_x, text_y = mid_point
            offset = 0.09  # Ajustement pour le texte pour le positionner légèrement au-dessus de l'arc
            text_x += offset * (pos[v][1] - pos[u][1])
            text_y += offset * (pos[u][0] - pos[v][0])

            # Ajouter le poids sur l'arc avec un fond blanc
            ax.text(
                text_x, text_y, f"{weight:.2f}", 
                fontsize=10, color="black", fontweight="bold",
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2")
            )
        # Ajouter des labels pour les arêtes
        #edge_labels = nx.get_edge_attributes(G, 'weight')
        #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        ax.set_title(f"CHOCO-GOSSIP : Iteration {frame}/{len(x_history)-1}")
        ax.text(0.5, -0.1, f"Current average is {avg_value:.2f}", transform=ax.transAxes, fontsize=12, ha='center')

    ani = FuncAnimation(fig, update, frames=range(0, len(x_history), int(len(x_history)/100)), interval=5, repeat=False)
    ani.save("./algos/results/animation_gossip_quantifie_choco.gif", dpi=80, writer="ffmpeg")
    plt.show()
