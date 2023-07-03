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

        scaler = pickle.load(open('../scalers/skl_subroutes.pkl', 'rb'))
        
        features = data[["day_of_week", "exit_time", "distance", "exit_stop", "target_stop"]]

        print(features)

        features = scaler.transform(features)

        labels = data['label'].values
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)

        model = load_model('../models/ANN_short_135.h5')
     
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

        # Plotting predicted time
        plt.scatter(X_test[:, 1], y_pred, c='b', label='Predicted Time', s=10, alpha=0.5)

        # Plotting actual time
        plt.scatter(X_test[:, 1], y_test, c='r', label='Actual Time', s=1, alpha=0.5)

        plt.xlabel('Exit Time')
        plt.ylabel('Arrive Time')
        plt.legend()
        plt.title('Exot time vs Predicted and Actual Arrive Time')

        # Showing only one dot per plot
        plt.legend(scatterpoints=1, frameon=False, labelspacing=1)
        plt.show()

    except Exception as e:
        handle_error(e)