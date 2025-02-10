
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture  import GaussianMixture
import matplotlib.pyplot as plt



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

bic_values = []
aic_values = []
n_componentsarr = []
for n in range(1, 16):
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X_scaled)
    n_componentsarr.append(n)
    bic_values.append(gmm.bic(X_scaled))
    aic_values.append(gmm.aic(X_scaled))
    print(f"n_components={n}, BIC={gmm.bic(X_scaled)}, AIC={gmm.aic(X_scaled)}")



plt.figure(figsize=(10, 6))

# Plot BIC
plt.plot(n_componentsarr, bic_values, marker='o', label='BIC')

# Plot AIC
plt.plot(n_componentsarr, aic_values, marker='o', label='AIC')

plt.xlabel('Number of Components')
plt.ylabel('Criterion Value')

# Add title and legend
plt.title('BIC and AIC vs Number of Components')

plt.show()



gmm = GaussianMixture(n_components=14, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

# Add cluster labels to the original dataset
data['gmm_cluster'] = gmm_labels

plt.figure(figsize=(12, 6))
plt.scatter(data['lon'], data['lat'], c=data['gmm_cluster'], cmap='viridis', s=50)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('GMM Clustering Results')
plt.colorbar(label='Cluster')
plt.show()

import plotly.express as px

# Create a scatter plot on a world map
fig = px.scatter_geo(
    data,
    lat='lat',
    lon='lon',
    color='gmm_cluster',
    color_continuous_scale='viridis',
    projection='natural earth',  # World map projection
    title='Earthquake Clusters (K-Means)',
    labels={'lat': 'Latitude', 'lon': 'Longitude', 'gmm_cluster': 'Cluster'}
)

# Show the plot
fig.show()


