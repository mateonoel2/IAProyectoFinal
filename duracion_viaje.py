import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_parquet("dataset")

# Calculate travel duration based on exit_time and label
data['duration'] = data['label'] - data['exit_time']

# Calculate average travel duration based on distance
avg_duration = data.groupby('distance')['duration'].mean()

# Create a line plot
plt.figure(figsize=(10, 6))
sns.lineplot(x=avg_duration.index, y=avg_duration.values)
plt.xlabel('Distancia (m)')
plt.ylabel('Tiempo de viaje promedio (s)')
plt.title('Tiempo de viaje promedio vs distancia')
plt.show()

data['duration_total'] = data['label'] - data['first_time']
#elimina todos los casos donde el tiempo de viaje es menor a 60 y mayor a 6000
data = data[(data['duration_total'] > 60) & (data['duration_total'] < 6000)]


avg_duration = data.groupby('total_distance')['duration_total'].mean()

# Create a line plot
plt.figure(figsize=(10, 6))
sns.lineplot(x=avg_duration.index, y=avg_duration.values)
plt.xlabel('Distancia total (m)')
plt.ylabel('Tiempo de viaje promedio (s)')
plt.title('Tiempo de viaje promedio vs distancia total')
plt.show()
