from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load dataset (Replace with actual file path)
df = pd.read_csv("~/Downloads/isc-gem-cat.csv")
df.columns = df.columns.str.strip()

# Selecting relevant columns
data = df[['lat', 'lon', 'depth', 'unc']]
data = data.loc[:, ~data.columns.duplicated(keep='first')]
data = data.dropna()

# Incorporate uncertainty into depth
data['adjusted_depth'] = data['depth'] / (1 + data['unc'])

# Selecting final features for clustering
X = data[['lat', 'lon', 'adjusted_depth']]

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Step 1: Determine eps using KNN plot
neighbors = NearestNeighbors(n_neighbors=5)  # min_samples=5
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

distances = np.sort(distances[:, -1])  # Sort by farthest neighbor distance
plt.plot(distances)
plt.xlabel('Data Points')
plt.ylabel('KNN Distance')
plt.title('KNN Distance Plot for eps Selection')
plt.show()

# Step 2: Fit DBSCAN with chosen eps and min_samples
dbscan = DBSCAN(eps=0.12, min_samples=5)  # Adjust eps based on KNN plot
labels = dbscan.fit_predict(X_scaled)

# Step 3: Add cluster labels to data and visualize
data['cluster'] = labels

# Visualize clusters (example with Matplotlib)
plt.scatter(data['lon'], data['lat'], c=data['cluster'], cmap='viridis', s=10)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('DBSCAN Clustering Results')
plt.colorbar(label='Cluster')
plt.show()




import plotly.express as px

# Create a scatter plot on a world map
fig = px.scatter_geo(
    data,
    lat='lat',
    lon='lon',
    color='cluster',
    color_continuous_scale='viridis',
    projection='natural earth',  # World map projection
    title='Earthquake Clusters (K-Means)',
    labels={'lat': 'Latitude', 'lon': 'Longitude', 'cluster': 'Cluster'}
)

# Show the plot
fig.show()

