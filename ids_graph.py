from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, lit, floor, rand
from graphframes import GraphFrame

# 1. Initialize Spark
spark = SparkSession.builder \
    .appName("IDS_Graph_Phase") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.4-s_2.12") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("--- Phase 2: Graph Construction (Simulated IPs) ---")

# 2. Load Pre-Cleaned Data
print("Loading pre-cleaned Parquet data...")
df = spark.read.parquet("cleaned_data.parquet")

print("Dataset is anonymized. Generating synthetic Source/Dest IPs for graph demo...")

df_graph = df.withColumn("src_octet", floor(rand() * 255).cast("string")) \
             .withColumn("dst_octet", floor(rand() * 255).cast("string")) \
             .withColumn("Source IP", concat(lit("192.168.1."), col("src_octet"))) \
             .withColumn("Destination IP", concat(lit("192.168.1."), col("dst_octet")))

# 3. Prepare Vertices and Edges
print("Extracting Graph Components...")

# Vertices: distinct IPs
src_ips = df_graph.select(col("Source IP").alias("id"))
dst_ips = df_graph.select(col("Destination IP").alias("id"))
vertices = src_ips.union(dst_ips).distinct()

# Edges: Src -> Dst (weighted by Flow Duration)
edges = df_graph.select(
    col("Source IP").alias("src"), 
    col("Destination IP").alias("dst"), 
    col("Flow_Duration").alias("duration")
)

# 4. Build Graph
g = GraphFrame(vertices, edges)

print(f"Total Vertices: {g.vertices.count()}")
print(f"Total Edges: {g.edges.count()}")

# 5. Run PageRank
print("--- Running PageRank (identifying critical nodes) ---")
# Reset probability 0.15, maxIter 5 (speed up for demo)
results = g.pageRank(resetProbability=0.15, maxIter=5)

print("Top 10 Critical IPs (Simulated):")
results.vertices.orderBy(col("pagerank").desc()).show(10, truncate=False)

# 6. Save Graph Features for Phase 3 (Ensemble)
# We need to join these PageRank scores back to the main data for the ML classifier
print("Saving Graph Features...")
# Save simple CSV of IP -> PageRank
results.vertices.write.mode("overwrite").csv("graph_features_pagerank")
print("Graph features saved to 'graph_features_pagerank' folder.")

print("Phase 2 Complete. Spark Session kept alive for 1 hour to view DAG at localhost:4040.")

import time
time.sleep(3600)
