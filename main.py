#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
df = pd.read_csv("Mall_Customers.csv")
x = df.iloc[:, 3:].values

#Using the elbow method to find the optimum number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state=42)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
#Here we found that the optimum number of clusters was 5

#Training the Kmeans Model
final = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = final.fit_predict(x)
print(y_kmeans)


#Visualizing the clusters
plt.scatter(x[y_kmeans==0,0], x[y_kmeans==0,1], s=100, c='red', label = 'cluster 1')
plt.scatter(x[y_kmeans==1,0], x[y_kmeans==1,1], s=100, c='blue', label = 'cluster 2')
plt.scatter(x[y_kmeans==2,0], x[y_kmeans==2,1], s=100, c='green', label = 'cluster 3')
plt.scatter(x[y_kmeans==3,0], x[y_kmeans==3,1], s=100, c='cyan', label = 'cluster 4')
plt.scatter(x[y_kmeans==4,0], x[y_kmeans==4,1], s=100, c='magenta', label = 'cluster 5')
plt.scatter(final.cluster_centers_[:,0], final.cluster_centers_[:,1], s=300, c='yellow', label = 'centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()
