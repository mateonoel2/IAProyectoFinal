from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import abs, asc, col
import sys

def handle_error(e):
    print("Error occurred during model training: ", e)
    sys.exit(1)

if __name__ == '__main__':    
    try:
        spark = SparkSession.builder \
        .appName("PySpark Linear Regression")\
        .config("spark.executor.instances", "4")\
        .config("spark.executor.cores", "1")\
        .config("spark.driver.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .config("spark.executor.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .getOrCreate()
        
        data = spark.read.parquet("../dataset").coalesce(1)\
                .orderBy(asc("month"), asc("day_of_month"), asc("exit_time")).cache()
        
        data = data.select("day_of_week", "exit_time", "distance", "exit_stop", "target_stop", "label")

        data.show(5)
        print("Number of rows:", data.count())

        assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features_a")
        data = assembler.transform(data).select("features_a", "label") 

        scaler = StandardScaler(inputCol="features_a", outputCol="features")
        scaler_model = scaler.fit(data)
        data = scaler_model.transform(data).select("features", "label")

        scaler_model.save("../scalers/spark_subroutes")

        split_point = int(data.count() * 0.8)

        train_data = data.limit(split_point).cache() 
        test_data = data.subtract(train_data).cache()  

        regressor = LinearRegression(featuresCol="features", labelCol="label")

        model = regressor.fit(train_data)

        predictions = model.transform(test_data)

        predictions = predictions.withColumn("abs_diff", abs(predictions["label"] - predictions["prediction"]))

        total_count = predictions.count()

        ranges = [600, 300, 60, 30]

        for r in ranges:
            within_range_count = predictions.filter(predictions["abs_diff"] <= r).count()
            percent_within_range = within_range_count / total_count * 100
            print("{:.2f}% of data is within a range of {:.0f} seconds".format(percent_within_range, r))

        evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        print("Root Mean Squared Error (RMSE) = {:.4f}".format(rmse))

        evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
        r2 = evaluator.evaluate(predictions)
        print("R-squared (R2) = {:.4f}\n".format(r2))

        model.save("../models/linear_regression_subroutes")

    except Exception as e:
        handle_error(e)
    finally:
        spark.stop()