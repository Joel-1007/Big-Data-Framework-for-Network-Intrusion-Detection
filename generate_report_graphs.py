import matplotlib.pyplot as plt
import networkx as nx
import random

def draw_attack_topology():
    print("Generating Graph 1: Attack Topology (Red Botnet)...")
    G = nx.DiGraph()
    
    # Nodes
    victim = "192.168.10.50"
    G.add_node(victim, color='red', size=3000, label="VICTIM\n(Web Server)")
    
    # Attackers (Botnet)
    attackers = [f"203.0.113.{i}" for i in range(1, 16)]
    for atk in attackers:
        G.add_node(atk, color='black', size=1000, label="Bot")
        G.add_edge(atk, victim, color='red', weight=2)

    # Benign Users (Green)
    users = ["192.168.1.5", "192.168.1.9"]
    for user in users:
        G.add_node(user, color='green', size=1000, label="User")
        G.add_edge(user, victim, color='green', weight=1)

    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw
    plt.figure(figsize=(10, 8))
    
    # Draw Nodes
    node_colors = [nx.get_node_attributes(G, 'color')[n] for n in G.nodes()]
    node_sizes = [nx.get_node_attributes(G, 'size')[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    
    # Draw Labels
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold')
    
    # Draw Edges
    edge_colors = [nx.get_edge_attributes(G, 'color')[e] for e in G.edges()]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrowstyle='-|>', arrowsize=20)
    
    plt.title("Fig 6.1: Attack Topology (Botnet Swarm)", fontsize=15)
    plt.axis('off')
    plt.savefig("attack_topology.png", format="PNG", dpi=300)
    print("Saved 'attack_topology.png'")

def draw_port_analysis():
    print("Generating Graph 2: Port Target Analysis...")
    G = nx.DiGraph()
    
    # Port Nodes
    ports = {"Port 80 (HTTP)": "blue", "Port 22 (SSH)": "orange"}
    for p, c in ports.items():
        G.add_node(p, color=c, size=3000)
        
    # Attackers grouping
    for i in range(1, 11):
        atk = f"Bot_{i}"
        target = "Port 80 (HTTP)" if i > 3 else "Port 22 (SSH)"
        col = "blue" if i > 3 else "orange"
        G.add_node(atk, color='gray', size=500)
        G.add_edge(atk, target, color=col)

    pos = nx.shell_layout(G)
    
    plt.figure(figsize=(10, 6))
    
    # Draw
    node_colors = [G.nodes[n].get('color', 'gray') for n in G.nodes()]
    node_sizes = [G.nodes[n].get('size', 500) for n in G.nodes()]
    edge_colors = [G.edges[e]['color'] for e in G.edges()]
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, 
            edge_color=edge_colors, font_weight='bold')
    
    plt.title("Fig 6.2: Port Service Targeting", fontsize=15)
    plt.savefig("port_analysis.png", format="PNG", dpi=300)
    print("Saved 'port_analysis.png'")

if __name__ == "__main__":
    try:
        draw_attack_topology()
        draw_port_analysis()
        print("\nSUCCESS: Graphs generated. Add them to your report.")
    except ImportError:
        print("Error: Missing libraries. Run 'pip install networkx matplotlib'")
