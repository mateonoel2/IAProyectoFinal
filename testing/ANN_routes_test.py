import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model

# Define a function to handle errors during model training
def handle_error(e):
    print("Error occurred during model training: ", e)
    sys.exit(1)

if __name__ == '__main__':    
    try:
        data = pd.read_parquet("../dataset") 

        scaler = pickle.load(open('../scalers/skl_routes.pkl', 'rb'))
        
        features = data[["day_of_week", "first_time", "total_distance", "first_stop", "target_stop"]]

        print(features)

        features = scaler.transform(features)

        labels = data['label'].values
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)

        model = load_model('../models/ANN_336.h5')
     
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Define the time thresholds in the range
        ranges = [600, 300, 60, 30]
        
        y_pred = y_pred.flatten()

        for r in ranges:    
            diff = np.abs(y_pred - y_test)
            within_range = diff <= r
            values_within_range = diff[within_range]
            percent_within_range = np.sum(within_range) / y_pred.shape[0] * 100
            print("{:.2f}% of data is within a range of {:.0f} seconds".format(percent_within_range, r))

        # Print metrics
        print("R-squared: {:.4f}".format(r2))
        print("RMSE: {:.4f}".format(rmse))

        #unscale x:
        X_test = scaler.inverse_transform(X_test)

        unique_x = np.unique(X_test[:, 2])
        averages = []
        for x in unique_x:
            indices = np.where(X_test[:, 2] == x)
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