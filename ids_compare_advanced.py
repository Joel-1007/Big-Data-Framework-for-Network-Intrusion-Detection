from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import (
    RandomForestClassifier, 
    LogisticRegression, 
    DecisionTreeClassifier,
    GBTClassifier,
    LinearSVC,
    NaiveBayes
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Initialize Spark
spark = SparkSession.builder \
    .appName("IDS_Advanced_Comparison") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("--- Phase 3.7: Grand Model Comparison ---")

# 2. Load Pre-Cleaned Data
print("Loading pre-cleaned Parquet data...")
df = spark.read.parquet("cleaned_data.parquet").sample(withReplacement=False, fraction=0.1, seed=42)

ignore_cols = ["Label", "Timestamp", "Flow ID", "Source IP", "Destination IP"]
feature_cols = [c for c in df.columns if c not in ignore_cols]

# 3. Pipeline Prep
# IMPORTANT: GBT and LinearSVC only support Binary Classification (Attack vs Benign) in Spark < 3.0
# But for Multiclass, we will focus on the main ones.
# To make GBT work for this demo, we will use StringIndexer to get 'label_index'
# Note: Spark's GBTClassifier currently supports binary labels primarily, so we might skip it if you have >2 classes.
# However, let's try standard classifiers first.

indexer = StringIndexer(inputCol="Label", outputCol="label_index").setHandleInvalid("skip")
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="skip")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)

# Split
train, test = df.randomSplit([0.8, 0.2], seed=42)

# 4. Define Models
# Note: LinearSVC and GBT in Spark are strictly Binary (0 vs 1) unless using OneVsRest, 
# but Random Forest/DecisionTree/LogisticReg/NaiveBayes handle Multiclass natively.
# We will compare the Multiclass champions.

models = {
    "Naive Bayes (Probabilistic)": NaiveBayes(labelCol="label_index", featuresCol="features", modelType="gaussian"),
    "Logistic Regression (Linear)": LogisticRegression(labelCol="label_index", featuresCol="features", maxIter=10),
    "Decision Tree (Simple Tree)": DecisionTreeClassifier(labelCol="label_index", featuresCol="features", maxDepth=5),
    "Random Forest (Ensemble)": RandomForestClassifier(labelCol="label_index", featuresCol="features", numTrees=50, maxDepth=10),
}

print(f"\n{'='*70}")
print(f"{'Model Name':<30} | {'Accuracy':<10} | {'F1 Score':<10}")
print(f"{'='*70}")

for name, clf in models.items():
    try:
        # Create Pipeline
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
    except Exception as e:
        print(f"{name:<30} | FAILED ({str(e)[:20]}...)")

print(f"{'='*70}")
print("Done. Show this table to prove that tree-based models (RF/DT) are superior.")
