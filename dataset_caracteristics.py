import pandas as pd

data = pd.read_parquet("dataset")

# Combine unique values from 'target_stop' and 'exit_stop'
all_stops = pd.concat([data['target_stop'], data['exit_stop']], ignore_index=True)

# Number of unique stops
num_stops = all_stops.nunique()

#Number of unque vahicles
num_vehicles = data['vehicle_id'].nunique()

# Number of unique first_stops
num_first_stops = data['first_stop'].nunique()

# Minimum and maximum distances
min_distance = data['distance'].min()
max_distance = data['distance'].max()

# Maximum total distance
max_total_distance = data['total_distance'].max()

# Minimum exit time and maximum arrive time
min_exit_time = data['exit_time'].min()
max_arrive_time = data['label'].max()

# Number of rows in the dataset
num_rows = len(data)

# Print the results
print("Number of vehicles:", num_vehicles)
print("Number of stops:", num_stops)
print("Number of first_stops:", num_first_stops)
print("Minimum distance:", min_distance)
print("Maximum distance:", max_distance)
print("Maximum total distance:", max_total_distance)
print("Minimum exit time:", min_exit_time)
print("Maximum arrive time:", max_arrive_time)
print("Number of rows:", num_rows)