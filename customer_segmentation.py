import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
    'Age':[20,25,30,35,40,45,50,55],
    'Income':[20000,25000,30000,50000,60000,65000,70000,80000]
}

df = pd.DataFrame(data)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Age','Income']])

print(df)

plt.scatter(df['Age'], df['Income'], c=df['Cluster'])
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Customer Segmentation')
plt.show()
