from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Initialize Spark
spark = SparkSession.builder \
    .appName("IDS_Model_Comparison") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("--- Phase 3.5: Model Comparison & Validation ---")

# 2. Load & Clean Data (Same robust cleaning as before)
print("Loading data...")
df = spark.read.csv("*.csv", header=True, inferSchema=True).sample(withReplacement=False, fraction=0.1, seed=42)
new_columns = [c.strip() for c in df.columns]
df = df.toDF(*new_columns)

# Clean NaN/Infinity
ignore_cols = ["Label", "Timestamp", "Flow ID", "Source IP", "Destination IP"]
feature_cols = [c for c in df.columns if c not in ignore_cols]

for c in feature_cols:
    df = df.withColumn(c, col(c).cast("double"))
    df = df.withColumn(c, when(isnan(col(c)) | (col(c) == float("inf")) | (col(c) == float("-inf")), 0.0).otherwise(col(c)))

df = df.na.fill(0.0)

# 3. Prepare Pipeline Components
indexer = StringIndexer(inputCol="Label", outputCol="label_index").setHandleInvalid("skip")
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="skip")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)

# Split Data
train, test = df.randomSplit([0.8, 0.2], seed=42)

# 4. Define the Models to Compare
models = {
    "Logistic Regression (Baseline)": LogisticRegression(labelCol="label_index", featuresCol="features", maxIter=10),
    "Decision Tree (Simple)": DecisionTreeClassifier(labelCol="label_index", featuresCol="features", maxDepth=5),
    "Random Forest (Ensemble)": RandomForestClassifier(labelCol="label_index", featuresCol="features", numTrees=50, maxDepth=10)
}

results = {}

# 5. Train and Evaluate Loop
print(f"\n{'='*60}")
print(f"{'Model Name':<30} | {'Accuracy':<10} | {'F1 Score':<10}")
print(f"{'='*60}")

best_model = None
best_predictions = None
best_name = ""

for name, clf in models.items():
    # Create Pipeline for this model
    pipeline = Pipeline(stages=[indexer, assembler, scaler, clf])
    
    # Train
    model = pipeline.fit(train)
    predictions = model.transform(test)
    
    # Evaluate
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="f1")
    
    acc = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    
    print(f"{name:<30} | {acc:.4f}     | {f1:.4f}")
    
    # Save the RF model for detailed analysis
    if "Random Forest" in name:
        best_model = model
        best_predictions = predictions
        best_name = name

print(f"{'='*60}")

# 6. The "Professor's Proof" (Confusion Matrix)
print(f"\n--- Detailed Analysis for {best_name} ---")
print("Confusion Matrix (Actual vs Predicted):")
# This shows EXACTLY what the model is getting right/wrong
# 0.0 = BENIGN (Normal), 1.0/2.0+ = ATTACKS
best_predictions.groupBy("label_index", "prediction").count().orderBy("label_index", "prediction").show(20)

print("Look at the rows where label_index > 0. These are the ATTACKS.")
print("If 'prediction' matches 'label_index', the model successfully caught the attack.")

# 7. Show actual Attack Detections
print("\nSample of SUCCESSFULLY DETECTED ATTACKS (Non-Zero predictions):")
best_predictions.filter("prediction > 0.0 AND label_index == prediction").select("Label", "label_index", "prediction").show(5)

print("Comparison Complete.")
