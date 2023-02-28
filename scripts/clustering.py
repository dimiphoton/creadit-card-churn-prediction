import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from preprocessing import preprocess_data

# load the data and preprocess it
#df = pd.read_csv('my_dataset.csv')
X_train, X_test, y_train, y_test = preprocess_data(df)

# create a range of k values to test
k_values = range(2, 10)

# evaluate the models using the silhouette score
silhouette_scores = []
for k in k_values:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_train)
    labels = model.predict(X_train)
    score = silhouette_score(X_train, labels)
    silhouette_scores.append(score)

# plot the silhouette scores for each k value
plt.plot(k_values, silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.show()

# choose the best value of k and fit the model
best_k = k_values[silhouette_scores.index(max(silhouette_scores))]
model = KMeans(n_clusters=best_k, random_state=42)
model.fit(X_train)
labels = model.predict(X_train)

# plot the clusters
plt.scatter(X_train[:, 0], X_train[:, 1], c=labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustering Results (k={})'.format(best_k))
plt.show()
