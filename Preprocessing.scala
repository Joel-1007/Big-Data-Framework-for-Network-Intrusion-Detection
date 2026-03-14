import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object Preprocessing {
  def main(args: Array[String]): Unit = {

    // 1. Initialize Spark Session
    val spark = SparkSession.builder()
      .appName("IDS_Scala_Preprocessing")
      .master("local[*]") // Use all available cores
      .getOrCreate()

    // Import implicit conversions for 569Xilsnotation
    import spark.implicits._
    
    // Set Log Level to minimize noise
    spark.sparkContext.setLogLevel("ERROR")

    println("--- PHASE 1 (SCALA): DATA INGESTION & CLEANING ---")

    // 2. Ingest Data (Reading CSV)
    val inputPath = "/Users/joeljohn/Desktop/MachineLearningCVE/*.csv" 
    println(s"Reading data from: $inputPath")

    var df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(inputPath)

    // 3. Sanitization: Removing leading/trailing spaces from column headers
    val cleanedColumns = df.columns.map(colName => colName.trim.replaceAll("\\s+", "_"))
    df = df.toDF(cleanedColumns: _*)
    println("Headers sanitized.")

    // 4. Error Handling: Replacing NaN and Infinity
    // Identify numeric columns (Double/Float/Integer)
    val numericTypes = Seq(DoubleType, FloatType, IntegerType, LongType)
    val numericCols = df.schema.fields.filter(f => numericTypes.contains(f.dataType)).map(_.name)

    println(s"Cleaning ${numericCols.length} numeric columns for Infinity/NaN...")

    numericCols.foreach { colName =>
      df = df.withColumn(colName, 
        when(col(colName).isNaN, 0.0) // Handle NaN
        .when(col(colName) === Double.PositiveInfinity, 0.0) // Handle +Infinity
        .when(col(colName) === Double.NegativeInfinity, 0.0) // Handle -Infinity
        .otherwise(col(colName)) // Keep original value if valid
      )
    }
    
    // Also fill standard NULLs with 0.0
    df = df.na.fill(0.0)

    // 5. Validation & Output
    println("--- PREPROCESSING COMPLETE ---")
    println(s"Total Flow Count: ${df.count()}")
    df.printSchema()
    
    // Save cleaned data to disk so Python scripts can use it
    val outputPath = "/Users/joeljohn/Desktop/MachineLearningCVE 2/cleaned_data.parquet"
    println(s"Saving cleaned data to: $outputPath")
    df.write.mode("overwrite").parquet(outputPath)
    
    spark.stop()
  }
}
