import sys
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Dense

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

        X_test, X_final_test, y_test, y_final_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)
        
        # Create a Sequential model
        model = Sequential()

        # Add the first layer with 49 neurons and 'relu' activation function
        model.add(Dense(49, activation='relu'))

        # Add the second layer with 49 neurons and 'relu' activation function
        model.add(Dense(49, activation='relu'))

        # Add the third layer with 49 neurons and 'relu' activation function
        model.add(Dense(49, activation='relu'))

        # Add the output layer with 'linear' activation function
        model.add(Dense(1, activation='linear'))

        # Compile the model with 'adam' optimizer, 'mean_squared_error'
        model.compile(optimizer='adam', loss='mean_squared_error')

        total_epochs = 500

        min_loss = np.inf
        for epoch in range(1, total_epochs+1):
            # Train the model
            model.fit(X_train, y_train, epochs=1, batch_size=256)

            # Evaluate the model
            loss = model.evaluate(X_test, y_test)

            if loss < min_loss:
                min_loss = loss
                print('mse:', min_loss)
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
                
                #Save best model 
                model.save(f'../models/ANN_short_{epoch}.h5')

                y_pred = model.predict(X_final_test)

                r2 = r2_score(y_final_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_final_test, y_pred))

                # Define the time thresholds in the range
                ranges = [600, 300, 60, 30]
                
                y_pred = y_pred.flatten()

                for r in ranges:    
                    diff = np.abs(y_pred - y_final_test)
                    within_range = diff <= r
                    values_within_range = diff[within_range]
                    percent_within_range = np.sum(within_range) / y_pred.shape[0] * 100
                    print("{:.2f}% of data is within a range of {:.0f} seconds".format(percent_within_range, r))

                # Print metrics
                print("R-squared: {:.4f}".format(r2))
                print("RMSE: {:.4f}".format(rmse))

    except Exception as e:
        handle_error(e)