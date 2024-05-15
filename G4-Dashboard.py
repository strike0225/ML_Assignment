import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

# Dataset After Preprocessing
new_restaurant = pd.read_csv("new_restaurant.csv")

# Tab Title
st.set_page_config(page_title="Uber Eats USA Restaurant", page_icon=":bar_chart:")

# Web Title
st.title("Uber Eats USA Restaurant :green[Analysis] :tea: :coffee: :chart: :bar_chart:")

# Affinity Propagation
st.title("Affinity Propagation")

#Affinity Propagation dataset
AP_Scores = pd.read_csv("AP_Silhouette_Scores.csv")
AP_New_Scores = pd.read_csv("AP_Silhouette_Scores_New.csv")

# Plot 1
st.title("Silhouette Scores for Different Preference Values (Size=1000)")
st.write(AP_New_Scores)

plt.figure()
plt.plot(AP_New_Scores['Preference'], AP_New_Scores['Silhouette Score'], marker='o')
plt.grid(True)
plt.xlabel('Preference Value')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Preference Values (Size=1000)')

st.pyplot(plt)

# Fuzzy C-Means
st.title("Fuzzy C-Means")

# Fuzzy_C_Means dataset
FPC_Score = pd.read_csv("FPC_values.csv")
FPC_Plot = pd.read_csv("FPC_Plot.csv")
Cluster_Center = pd.read_csv("Cluster_Centers.csv")

# Plot 1
st.title('FPC vs Number of Clusters')
st.write(FPC_Score)

plt.figure()
plt.plot(FPC_Score['Number of Clusters'], FPC_Score['Fuzzy Partition Coefficient (FPC)'], marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Fuzzy Partition Coefficient (FPC)')
plt.title('FPC vs Number of Clusters')

st.pyplot(plt)

# Plot 2
data_transposed = FPC_Plot[['Feature 1', 'Feature 2']].values.T
cluster_membership = FPC_Plot['Cluster Membership'].values

plt.figure(figsize=(8, 6))
scatter = plt.scatter(data_transposed[0], data_transposed[1], c=cluster_membership, marker='o', cmap='viridis', alpha=0.7, edgecolors='k')
plt.scatter(Cluster_Center['Feature 1'], Cluster_Center['Feature 2'], c='red', marker='x', s=100, label='Cluster Centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Fuzzy C-Means Clustering with {len(Cluster_Center)} Clusters')
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster Membership')
plt.legend()

st.pyplot(plt)

# Spectral Clustering
st.title("Spectral Clustering")

# Spectral Clustering dataset
Spectral_Clustering = pd.read_csv("Spectral_Clustering.csv")
SC_Silhouette_Scores = pd.read_csv("SC_Silhouette_Scores.csv")
Best_Score = pd.read_csv("Best_Silhouette_Score.csv")

# Plot 1
st.title('Inertia for Different Number of Clusters')
st.write(Spectral_Clustering)

plt.figure(figsize=(10, 6))
plt.plot(Spectral_Clustering['Number of Clusters'], Spectral_Clustering['Inertia'], marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-cluster Sum of Squares')
plt.title('Inertia for Different Number of Clusters')

st.pyplot(plt)

# Plot 2
st.title('Silhouette Scores for Different Number of Clusters')
st.write(SC_Silhouette_Scores)

plt.figure(figsize=(10, 6))
plt.plot(SC_Silhouette_Scores['Number of Clusters'], SC_Silhouette_Scores['Silhouette Score'], marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Number of Clusters')

st.pyplot(plt)

# Plot 3
best_silhouette_score = SC_Silhouette_Scores['Silhouette Score'].max()
best_n_clusters_id = SC_Silhouette_Scores['Silhouette Score'].idxmax()
best_n_clusters = SC_Silhouette_Scores.loc[best_n_clusters_id, 'Number of Clusters']

st.write(f"Best Silhouette Score: {best_silhouette_score}")
st.write(f"Number of Clusters: {best_n_clusters}")

plt.figure(figsize=(8, 6))
scatter = plt.scatter(Best_Score['PCA1'], Best_Score['PCA2'], c=Best_Score['ClusterLabel'], cmap='viridis', edgecolors='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'Spectral Clustering (Best number of clusters: {best_n_clusters})')

cbar = plt.colorbar(scatter)
cbar.set_label('Cluster Label')

st.pyplot(plt)

# Gaussian Mixture Model (GMM)
st.title("Gaussian Mixture Model (GMM)")

# Gaussian Mixture Model (GMM) dataset
GMM_Score = pd.read_csv("GMM_Score.csv")
GMM_Clustering = pd.read_csv("GMM_Clustering.csv")

# Plot 1
st.title('Silhouette Score by Number of Clusters')
st.write(GMM_Score)

plt.figure(figsize=(10, 6))
plt.plot(GMM_Score['Number of Clusters'], GMM_Score['Silhouette Score'], marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by Number of Clusters')

st.pyplot(plt)

# Plot 2
best_num_clusters = GMM_Clustering['Cluster Label'].nunique()


plt.figure(figsize=(8, 6))
scatter = plt.scatter(GMM_Clustering['PC1'], GMM_Clustering['PC2'], c=GMM_Clustering['Cluster Label'], cmap='viridis', alpha=0.8, edgecolors='k')
plt.colorbar(label='Cluster')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'PCA of Clusters (Best Number of Clusters: {best_num_clusters})')

st.pyplot(plt)

# Hierarchical Clustering
st.title("Hierarchical Clustering")

# Hierarchical Clustering dataset
Linkage = pd.read_csv("Linkage_Matrix.csv")
Hiearchical_Score = pd.read_csv("Hiearchical_Score.csv")
Hiearchical_Clustering = pd.read_csv("Hiearchical_Clustering.csv")

# Plot 1
linkage_matrix = Linkage.to_numpy()

st.title('Hierarchical Clustering Dendrogram (Truncated)')

plt.figure(figsize=(12, 8))
dendrogram(
    linkage_matrix,
    leaf_rotation=45,
    leaf_font_size=6,
    truncate_mode='level',
    p=4
)

plt.title('Hierarchical Clustering Dendrogram (Truncated)')
plt.xlabel('Sample Index (or Cluster Size)')
plt.ylabel('Distance')

st.pyplot(plt)

# Plot 2
st.title('Silhouette Score vs. Number of Clusters')
st.write(Hiearchical_Score)

plt.figure(figsize=(8, 6))
plt.plot(Hiearchical_Score['Number of Clusters'], Hiearchical_Score['Silhouette Score'], marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')

st.pyplot(plt)

# Plot 3
best_n_clusters = Hiearchical_Clustering['Cluster'].nunique()
st.title(f'Agglomerative Clustering (Best Number of Clusters: {best_n_clusters})')

plt.figure(figsize=(8, 6))
for cluster_id in Hiearchical_Clustering['Cluster'].unique():
    cluster_points = Hiearchical_Clustering[Hiearchical_Clustering['Cluster'] == cluster_id]
    plt.scatter(cluster_points['PC1'], cluster_points['PC2'], label=f'Cluster {cluster_id}', edgecolors='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'Agglomerative Clustering (Best number of clusters: {best_n_clusters})')
plt.legend()

st.pyplot(plt)