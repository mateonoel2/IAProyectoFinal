from pyspark.sql import SparkSession
from pyspark.ml.regression import GBTRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.sql.functions import abs, asc
import numpy as np
import matplotlib.pyplot as plt
import sys

def handle_error(e):
    print("Error occurred during model training: ", e)
    sys.exit(1)

if __name__ == '__main__':    
    try:
        spark = SparkSession.builder \
        .config("spark.executor.instances", "4")\
        .config("spark.executor.cores", "1")\
        .config("spark.driver.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .config("spark.executor.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .getOrCreate()
        
        data = spark.read.parquet("../dataset").coalesce(1)\
                .orderBy(asc("month"), asc("day_of_month"), asc("exit_time")).cache()
        
        data = data.select("day_of_week", "first_time", "total_distance", "first_stop", "target_stop", "label")

        data.show(5)
    
        assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features_a")
        data = assembler.transform(data).select("features_a", "label") 

        scaler = StandardScalerModel.load("../scalers/spark_routes")
        data = scaler.transform(data).select("features", "label")

        split_point = int(data.count() * 0.8)

        train_data = data.limit(split_point).cache() 
        test_data = data.subtract(train_data).cache()  

        model = GBTRegressionModel.load("../models/gbt_routes")

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
    
        # Unscaled data and set X_test, y_pred, and y_test
        unscaled_test_data = test_data.select("features", "label").toPandas()

        X_test = np.array(unscaled_test_data["features"].apply(lambda x: x[2]).tolist())
        y_test = unscaled_test_data["label"].values

        predictions_pd = predictions.select("prediction").toPandas()
        y_pred = predictions_pd["prediction"].values


        unique_x = np.unique(X_test)
        averages = []
        for x in unique_x:
            indices = np.where(X_test == x)
            avg_y_pred = np.mean(y_pred[indices])
            avg_y_test = np.mean(y_test[indices])
            averages.append((x, avg_y_pred, avg_y_test))

        averages = np.array(averages)

        # Plotting the averaged points
        plt.scatter(averages[:, 0], averages[:, 1], c='b', label='Average Predicted Time', s=10, alpha=0.5)
        plt.scatter(averages[:, 0], averages[:, 2], c='r', label='Average Actual Time', s=5, alpha=0.5)

        # Adding lines connecting the points
        for i in range(len(averages)):
            x = averages[i, 0]
            y_pred = averages[i, 1]
            y_test = averages[i, 2]
            plt.plot([x, x], [y_pred, y_test], c='black', linewidth=0.5)

        plt.xlabel('Distance')
        plt.ylabel('Arrive Time')
        plt.legend()
        plt.title('Distance vs Average Predicted and Actual Arrive Time')

        plt.show()
        plt.clf()
       
        absolute_difference = np.abs(averages[:, 1] - averages[:, 2])

        # Plotting the averaged points
        plt.scatter(averages[:, 0], absolute_difference, c='g', label='Absolute Difference', s=10, alpha=0.5)
        plt.xlabel('Distance')
        plt.ylabel('Absolute Difference')
        plt.legend()
        plt.title('Distance vs Absolute Difference between Average Predicted and Actual Value')

        plt.show()

    except Exception as e:
        handle_error(e)
    finally:
        spark.stop()