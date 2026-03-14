import pandas as pd
from neo4j import GraphDatabase
import random
import glob
import os

# --- CONFIGURATION ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "test1234") 

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

def load_and_push_data():
    print("--- PHASE 1: LOADING CSV DATA ---")
    # 1. Read a sample of the CSV
    csv_files = glob.glob("*.csv")
    if not csv_files:
        print("ERROR: No CSV files found! Make sure you are in the MachineLearningCVE folder.")
        return

    print(f"Reading from: {csv_files[0]}...")
    df = pd.read_csv(csv_files[0])
    
    # Strip whitespace
    df.columns = df.columns.str.strip()
    
    # 2. Filter: Get 100 Attacks and 50 Benign flows
    try:
        attacks = df[df['Label'] != 'BENIGN'].sample(n=100, replace=True)
        benign = df[df['Label'] == 'BENIGN'].sample(n=50, replace=True)
    except ValueError:
        # Fallback if file is too small
        attacks = df[df['Label'] != 'BENIGN']
        benign = df[df['Label'] == 'BENIGN']

    combined = pd.concat([attacks, benign])
    print(f"Selected {len(combined)} flows for visualization.")

    # 3. Generate Synthetic Topology
    victim_ip = "192.168.10.50"
    topology_data = []
    
    for index, row in combined.iterrows():
        label = row['Label']
        port = row['Destination Port']
        
        # Synthetic IP Logic
        if label == 'BENIGN':
            src_ip = f"192.168.1.{random.randint(2, 254)}"
            dst_ip = f"192.168.1.{random.randint(2, 254)}"
            color = "green"
            type = "User"
        else:
            # Botnet targets Victim
            src_ip = f"203.0.{random.randint(1, 113)}.{random.randint(1, 255)}"
            dst_ip = victim_ip
            color = "red"
            type = "Attacker"

        topology_data.append({
            "src": src_ip,
            "dst": dst_ip,
            "label": label,
            "port": int(port),
            "color": color,
            "type": type
        })

    # 4. Push to Neo4j
    print("--- PHASE 2: PUSHING TO NEO4J ---")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n") # Clear old data
        session.run("CREATE (:IP {id: $ip, type: 'Victim'})", ip=victim_ip)
        
        count = 0
        for flow in topology_data:
            query = """
            MERGE (a:IP {id: $src})
            ON CREATE SET a.type = $type
            MERGE (b:IP {id: $dst})
            MERGE (a)-[r:FLOW {label: $label, dest_port: $port, color: $color}]->(b)
            """
            session.run(query, 
                        src=flow['src'], 
                        dst=flow['dst'], 
                        type=flow['type'],
                        label=flow['label'], 
                        port=flow['port'], 
                        color=flow['color'])
            count += 1
            if count % 20 == 0:
                print(f"Pushed {count} flows...")

    print("SUCCESS: Data uploaded. Go to http://localhost:7474 to visualize.")

if __name__ == "__main__":
    try:
        load_and_push_data()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Did you start the Docker container? (docker start neo4j)")
