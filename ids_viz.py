from pyspark.sql import SparkSession
from pyspark.sql.functions import col, floor, rand, concat, lit
from graphframes import GraphFrame
from neo4j import GraphDatabase
import time

# --- SPARK SECTION ---
print("--- Step 1: Computing Graph Metrics in Spark ---")
spark = SparkSession.builder \
    .appName("IDS_Viz_Export") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.4-s_2.12") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Generate Synthetic IPs
df = spark.read.csv("*.csv", header=True, inferSchema=True).sample(fraction=0.05, seed=42)
df = df.toDF(*[c.strip() for c in df.columns])

df_graph = df.withColumn("src_octet", floor(rand() * 50).cast("string")) \
             .withColumn("dst_octet", floor(rand() * 50).cast("string")) \
             .withColumn("src", concat(lit("192.168.1."), col("src_octet"))) \
             .withColumn("dst", concat(lit("192.168.1."), col("dst_octet"))) \
             .withColumn("weight", col("Flow Duration"))

# Build Graph & PageRank
edges = df_graph.select("src", "dst", "weight")
vertices = df_graph.select(col("src").alias("id")).union(df_graph.select(col("dst").alias("id"))).distinct()
g = GraphFrame(vertices, edges)

print("Running PageRank...")
ranks = g.pageRank(resetProbability=0.15, maxIter=5)

# Extract Top Nodes
top_nodes = ranks.vertices.orderBy(col("pagerank").desc()).limit(20)
top_ids = [row.id for row in top_nodes.collect()]
viz_edges = edges.filter(col("src").isin(top_ids) & col("dst").isin(top_ids)).limit(100).collect()

print(f"Prepared {len(top_ids)} nodes and {len(viz_edges)} edges for visualization.")

# --- NEO4J SECTION ---
print("--- Step 2: Pushing to Neo4j ---")

class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_db(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_graph(self, nodes, edges):
        with self.driver.session() as session:
            # Create Nodes
            for node_id in nodes:
                # The $addr variable will now be preserved correctly!
                session.run("MERGE (a:IPAddress {address: $addr})", addr=node_id)
            
            # Create Edges
            for row in edges:
                session.run("""
                    MATCH (a:IPAddress {address: $src}), (b:IPAddress {address: $dst})
                    MERGE (a)-[:SENT_FLOW {duration: $dur}]->(b)
                """, src=row.src, dst=row.dst, dur=float(row.weight))

try:
    loader = Neo4jLoader("bolt://localhost:7687", "neo4j", "project2025")
    loader.clear_db()
    loader.create_graph(top_ids, viz_edges)
    loader.close()
    print("SUCCESS: Graph exported to Neo4j!")
    print("OPEN BROWSER: http://localhost:7474")
    print("Login -> User: neo4j | Pass: project2025")
except Exception as e:
    print(f"Error connecting to Neo4j: {e}")
