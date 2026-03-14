import matplotlib.pyplot as plt
import networkx as nx
import random

# --- GRAPH 1: RED ATTACK TOPOLOGY ---
def draw_attack_topology():
    print("Generating Graph 1: Attack Topology (Red Botnet)...")
    G = nx.DiGraph()
    
    # Nodes
    victim = "192.168.10.50"
    G.add_node(victim, color='red', size=4000, label="VICTIM\n(Web Server)")
    
    # Attackers (Botnet)
    # Create a star shape manually
    attackers = [f"203.0.113.{i}" for i in range(1, 16)]
    for atk in attackers:
        G.add_node(atk, color='black', size=1500, label="Bot")
        G.add_edge(atk, victim, color='red', weight=2)

    # Benign Users (Green) - Just a few to show contrast
    users = ["192.168.1.5", "192.168.1.9"]
    for user in users:
        G.add_node(user, color='green', size=1500, label="User")
        G.add_edge(user, victim, color='green', weight=1)

    # Layout - Spring layout often gives a good "Star" burst
    pos = nx.spring_layout(G, k=0.6, seed=42)
    
    plt.figure(figsize=(12, 10))
    
    # Draw Nodes
    node_colors = [nx.get_node_attributes(G, 'color')[n] for n in G.nodes()]
    node_sizes = [nx.get_node_attributes(G, 'size')[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black')
    
    # Draw Labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=labels, font_color='white', font_weight='bold', font_size=9)
    
    # Draw Edges
    edge_colors = [nx.get_edge_attributes(G, 'color')[e] for e in G.edges()]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrowstyle='-|>', arrowsize=25, width=1.5)
    
    plt.title("Fig 6.1: Attack Topology (Botnet Swarm targeting Victim)", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.savefig("attack_topology.png", format="PNG", dpi=300, bbox_inches='tight')
    print("Saved 'attack_topology.png'")

# --- GRAPH 2: PORT TARGET ANALYSIS ---
def draw_port_analysis():
    print("Generating Graph 2: Port Target Analysis...")
    G = nx.DiGraph()
    
    # Target Ports
    ports = {"Port 80 (HTTP)": "blue", "Port 22 (SSH)": "orange"}
    for p, c in ports.items():
        G.add_node(p, color=c, size=5000, label=p)
        
    # Attackers grouping
    for i in range(1, 13):
        atk = f"Bot_{i}"
        # Split attack traffic: Most to HTTP, some to SSH
        if i > 4: 
            target = "Port 80 (HTTP)"
            col = "blue"
        else:
            target = "Port 22 (SSH)"
            col = "orange"
            
        G.add_node(atk, color='gray', size=800, label=atk)
        G.add_edge(atk, target, color=col)

    # Use a Shell Layout to clearly separate Layers
    pos = nx.shell_layout(G, nlist=[list(ports.keys()), [n for n in G.nodes() if "Bot" in n]])
    
    plt.figure(figsize=(10, 8))
    
    # Draw
    node_colors = [G.nodes[n].get('color', 'gray') for n in G.nodes()]
    node_sizes = [G.nodes[n].get('size', 800) for n in G.nodes()]
    edge_colors = [G.edges[e]['color'] for e in G.edges()]
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, 
            edge_color=edge_colors, font_weight='bold', font_color='black', font_size=10, arrowsize=20)
    
    plt.title("Fig 6.2: Port Service Targeting (HTTP vs SSH)", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.savefig("port_analysis.png", format="PNG", dpi=300, bbox_inches='tight')
    print("Saved 'port_analysis.png'")

if __name__ == "__main__":
    draw_attack_topology()
    draw_port_analysis()
    print("\nSUCCESS: Images generated in your folder.")
