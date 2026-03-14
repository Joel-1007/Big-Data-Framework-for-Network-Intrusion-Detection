from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType

# 1. Initialize Spark with Kafka Support
spark = SparkSession.builder \
    .appName("IDS_RealTime_Streaming") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("--- Phase 4: Real-Time Streaming Detection ---")

# 2. Define Schema (Simplified for Demo)
# We pick a few key columns to visualize
schema = StructType() \
    .add("Destination Port", StringType()) \
    .add("Flow Duration", StringType()) \
    .add("Total Fwd Packets", StringType()) \
    .add("Label", StringType())

# 3. Read Stream from Kafka
print("Connecting to Kafka Stream...")
df_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "network-flows") \
    .option("startingOffsets", "latest") \
    .load()

# 4. Parse JSON
parsed_stream = df_stream.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# 5. Add 'Alert' Logic
# In a real system, you would apply your ML Model here.
# For this demo, we just format the output to look like a dashboard.
dashboard_stream = parsed_stream.select(
    col("Label").alias("Traffic_Type"),
    col("Destination Port").alias("Dst_Port"),
    col("Flow Duration").alias("Duration_ms"),
    col("Total Fwd Packets").alias("Packets")
)

# 6. Start the Dashboard (Output to Console)
print("Starting Dashboard... (Waiting for data)")
query = dashboard_stream \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
