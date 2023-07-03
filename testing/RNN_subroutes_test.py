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
        
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)

        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        model = load_model('../models/RNN_short_27.h5')
     
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        ranges = [600, 300, 60, 30]
        
        y_pred = y_pred.flatten()

        for r in ranges:    
            diff = np.abs(y_pred - y_test)
            within_range = diff <= r
            values_within_range = diff[within_range]
            percent_within_range = np.sum(within_range) / y_pred.shape[0] * 100
            print("{:.2f}% of data is within a range of {:.0f} seconds".format(percent_within_range, r))

        print("R-squared: {:.4f}".format(r2))
        print("RMSE: {:.4f}".format(rmse))

    except Exception as e:
        handle_error(e)