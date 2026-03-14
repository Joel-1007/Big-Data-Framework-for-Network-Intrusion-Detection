from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, isnan
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, Imputer, VectorAssembler, StringIndexer

# Initialize Spark
spark = SparkSession.builder \
    .appName("IDS_Classifier_Project") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.4-s_2.12") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("--- Phase 1: Data Ingestion ---")
# Load Data
df = spark.read.csv("*.csv", header=True, inferSchema=True)

# 1. Clean Column Names (Remove leading/trailing spaces)
new_columns = [c.strip() for c in df.columns]
df = df.toDF(*new_columns)
print(f"Data Loaded. Rows: {df.count()}")

# 2. Fix Labels (Handle minor naming inconsistencies if any)
# We want to separate 'Label' from the features
label_col = "Label"
feature_cols = [c for c in df.columns if c != label_col]

print(f"Identified {len(feature_cols)} feature columns.")

# 3. Handle Types & Infinity
# Convert all features to double (required for scaling)
for c in feature_cols:
    df = df.withColumn(c, col(c).cast("double"))

# Drop rows with minimal nulls (usually <0.1% in this dataset)
df_clean = df.na.drop(subset=feature_cols)

print(f"Rows after dropping nulls: {df_clean.count()}")

print("--- Phase 2: Feature Engineering Pipeline ---")

# 4. Vector Assembler -> Imputer -> Scaler
# Assemble all features into a single vector column 'features_raw'
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="skip")

# Standard Scaler (Normalize data)
scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True)

# String Indexer for Label (Converts 'BENIGN', 'DDoS' to 0, 1, 2...)
indexer = StringIndexer(inputCol="Label", outputCol="label_index")

# Build the Pipeline
pipeline = Pipeline(stages=[assembler, scaler, indexer])

print("Fitting pipeline (this may take a minute)...")
model = pipeline.fit(df_clean)
processed_df = model.transform(df_clean)

print("--- Preprocessing Complete ---")
processed_df.select("features_scaled", "label_index").show(5, truncate=False)

# Save a small sample for verification
processed_df.select("features_scaled", "label_index").limit(100).write.mode("overwrite").parquet("processed_sample.parquet")
print("Sample saved to 'processed_sample.parquet'")
