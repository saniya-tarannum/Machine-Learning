import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

# Load the data
df = pd.read_csv('mall-customers-data.csv') 

# Verify column names
print(df.columns)

# Extract relevant features (ensure column names match your dataset)
X = df[['annual_income', 'spending_score']].values  

# Calculate Sum of Squared Errors (SSE) for each number of clusters (k) from 1 to 10
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=23)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)  # inertia_ is the SSE for the given k

# Plotting the Elbow Method using Seaborn
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
sns.lineplot(x=range(1, 11), y=sse, marker='o')

# Setting up plot aesthetics
plt.xlabel("Number of clusters (k)")
plt.ylabel("Sum of Squared Error (SSE)")
plt.title("Elbow Method for Optimal k")
plt.show()
