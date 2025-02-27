# import numpy as np

# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# from sklearn.preprocessing import StandardScaler

# # Load dataset (Replace with actual file path)
# df = pd.read_csv("~/Downloads/isc-gem-cat.csv")
# df.columns = df.columns.str.strip()

# # Selecting relevant columns
# data = df[['lat', 'lon', 'depth', 'unc']]
# data = data.loc[:, ~data.columns.duplicated(keep='first')]
# data = data.dropna()

# # Incorporate uncertainty into depth
# data['adjusted_depth'] = data['depth'] / (1 + data['unc'])

# # Selecting final features for clustering
# X = data[['lat', 'lon', 'adjusted_depth']]

# # Normalize data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Initialize DataFrame to store results
# results = []

# # Loop through k values from 6 to 20
# for k in range(6, 21):
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     clusters = kmeans.fit_predict(X_scaled)
    
#     # Compute clustering evaluation metrics
#     silhouette = silhouette_score(X_scaled, clusters)
#     calinski_harabasz = calinski_harabasz_score(X_scaled, clusters)
#     davies_bouldin = davies_bouldin_score(X_scaled, clusters)
    
#     # Store results
#     results.append([k, silhouette, calinski_harabasz, davies_bouldin])

# # Convert results into DataFrame
# results_df = pd.DataFrame(results, columns=['Clusters', 'Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Index'])

# # Display results
# print(results_df)

# # Plot the evaluation metrics
# fig, ax1 = plt.subplots(figsize=(10, 6))

# ax1.set_xlabel('Number of Clusters (k)')
# ax1.set_ylabel('Silhouette Score', color='tab:blue')
# ax1.plot(results_df['Clusters'], results_df['Silhouette Score'], marker='o', color='tab:blue', label='Silhouette Score')
# ax1.tick_params(axis='y', labelcolor='tab:blue')

# ax2 = ax1.twinx()
# ax2.set_ylabel('Davies-Bouldin Index (lower is better)', color='tab:red')
# ax2.plot(results_df['Clusters'], results_df['Davies-Bouldin Index'], marker='s', color='tab:red', linestyle='dashed', label='Davies-Bouldin Index')
# ax2.tick_params(axis='y', labelcolor='tab:red')

# fig.suptitle('K-Means Clustering Evaluation Metrics')
# fig.tight_layout()
# plt.show()

# # Choose optimal K (Manually select based on elbow plot)
# k_optimal = 10  # Adjust based on elbow curve observation

# # Apply K-Means clustering
# kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
# data['cluster'] = kmeans.fit_predict(X_scaled)

# # Compute clustering evaluation metrics
# silhouette = silhouette_score(X_scaled, data['cluster'])
# calinski_harabasz = calinski_harabasz_score(X_scaled, data['cluster'])
# davies_bouldin = davies_bouldin_score(X_scaled, data['cluster'])

# # Print results
# print(f'Silhouette Score: {silhouette:.4f}')
# print(f'Calinski-Harabasz Score: {calinski_harabasz:.4f}')
# print(f'Davies-Bouldin Index: {davies_bouldin:.4f}')

# # Visualizing clusters
# plt.figure(figsize=(8, 6))
# plt.scatter(data['lon'], data['lat'], c=data['cluster'], cmap='viridis', alpha=0.7)
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Earthquake Clusters (K-Means)')
# plt.colorbar(label='Cluster')
# plt.show()

# import plotly.express as px

# # Create a scatter plot on a world map
# fig = px.scatter_geo(
#     data,
#     lat='lat',
#     lon='lon',
#     color='cluster',
#     color_continuous_scale='viridis',
#     projection='natural earth',  # World map projection
#     title='Earthquake Clusters (K-Means)',
#     labels={'lat': 'Latitude', 'lon': 'Longitude', 'cluster': 'Cluster'}
# )

# # Show the plot
# fig.show()



from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd


# Load dataset (Replace with actual file path)
df = pd.read_csv("~/Downloads/isc-gem-cat.csv")
df.columns = df.columns.str.strip()

# Selecting relevant columns
data = df[['lat', 'lon', 'depth']]
data = data.loc[:, ~data.columns.duplicated(keep='first')]
data = data.dropna()


# Selecting final features for clustering
X = data[['lat', 'lon', 'depth']]

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
for k in range(6, 31):
    kmeans = KMeans(n_clusters=k, random_state=42,n_init=10)
    kmeans.fit(X_scaled)
    inertia.append([k, kmeans.inertia_])  # Store k and inertia as pairs

# Calculate differences between consecutive inertia values
inertia_diff = [inertia[i][1] - inertia[i+1][1] for i in range(len(inertia)-1)]
clusters_k = [inertia[i][0] for i in range(len(inertia)-1)]  # Exclude last k
inertia_k = [inertia[i][1] for i in range(len(inertia)-1)]  # Exclude last k

# Create DataFrame
inertia_diff_df = pd.DataFrame({'Clusters': clusters_k, 'inertia': inertia_k, 'Inertia_Diff': inertia_diff})





print("cluser quality using inertia method:" ,inertia_diff_df)

plt.figure(figsize=(12, 6))
plt.plot(inertia_diff_df['Clusters'], inertia_diff_df['Inertia_Diff'], marker='o', label='Inertia Difference')
plt.axhline(0, color='red', linestyle='--', label='No Improvement') # Horizontal line at y=0
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia Difference')
plt.title('Inertia Difference vs Number of Clusters')
plt.grid(True)
plt.legend()
plt.show()


# Choose optimal K (Manually select based on elbow plot)
k_optimal = 6  # Adjust based on elbow curve observation

# Apply K-Means clustering
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(X_scaled)



# Visualizing clusters
plt.figure(figsize=(8, 6))
plt.scatter(data['lon'], data['lat'], c=data['cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Earthquake Clusters (K-Means)')
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






