#!/usr/bin/env python
# coding: utf-8

# <div style="float: right; margin-left: 20px;">
#   <img src="logo.png" alt="Logo" width="105" height="105"/>
# </div>
# 
# Democratic and Popular Republic of Algeria \
# Ministry of Higher Education and Scientific Research \
# National Higher School of Computer Science - May 9 1945 - Sidi Bel Abbes \
# 
# ### Project: Customer Segmentation Using the <span style="color:blue;">K-means Algorithm</span> with <span style="color:blue;">Preprocessing</span> </span> - A Study
# ___
# #### Table of Contents
# 1. [Data Cleaning and Visualization](#data-cleaning-and-visualization)
# 2. [Feature Engineering](#feature-engineering)
# 3. [Data Preprocessing](#data-preprocessing)
# 4. [Feature Scaling](#feature-scaling)
# 5. [Computing Optimal K for K-means Clustering](#computing-optimal-k-for-clustering)
# 6. [Clustering our Dataset](#clustering-our-dataset)
# 7. [Evaluating our Model's Performance](#evaluating-our-model-performance)
# 8. [Conclusion](#conclusion)
# 
# More information about the dataset can be found [here](https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset).
# 
# <hr style="border: 0.3px solid black;"/>

# ## Importing the libraries

# In[1]:


import numpy    as      np
import pandas   as      pd
import seaborn  as      sns
from   scipy    import  stats

import matplotlib.pyplot   as      plt
from   matplotlib.patches  import  Ellipse

from sklearn.cluster        import  KMeans
from sklearn.metrics        import  silhouette_score
from sklearn.metrics        import  davies_bouldin_score
from sklearn.decomposition  import  PCA
from sklearn.preprocessing  import  MinMaxScaler
from sklearn.preprocessing  import  LabelEncoder


df = pd.read_csv("customer_shopping_data.csv")

def preprocess_and_format(df):
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y')

    df['day_of_week'] = df['invoice_date'].dt.dayofweek
    df['month'] = df['invoice_date'].dt.month
    df['year'] = df['invoice_date'].dt.year

    age_bins = [18, 25, 35, 50, 80]
    age_labels = ['19-25', '26-35', '36-50', '51+']
    df['age'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

    df['total_spending'] = df['quantity'] * df['price']

    df = df.drop(['invoice_no', 'customer_id'], axis=1)

    df['gender_age_group'] = df['gender'] + '_' + df['age'].astype(str)

    df = df[['category',
         'gender',
         'age',
         'gender_age_group',
         'payment_method',
         'shopping_mall',
         'invoice_date',
         'day_of_week',
         'month', 
         'year',
         'quantity',
         'price',
         'total_spending']]

    encoding_mapping = {}

    features_to_encode = ['category', 'age', 'gender', 'gender_age_group', 'payment_method', 'shopping_mall']

    label_encoder = LabelEncoder()

    for feature in features_to_encode:
        df[feature] = label_encoder.fit_transform(df[feature])
    
        # Store the mapping in the dictionary
        encoding_mapping[feature] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    X = df[['total_spending']]

    scaler = MinMaxScaler()

    X_normalized = scaler.fit_transform(X)

    df[['total_spending']] = X_normalized

    return df

def compute(df):
    X = df[['category',
        'payment_method',
        'gender_age_group',
        'shopping_mall',
        'total_spending']]

    inertia = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    X = df[['category',
        'gender_age_group',
        'payment_method',
        'shopping_mall',
        'total_spending']]

    num_clusters = 4

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    cluster_assignments = kmeans.fit_predict(X)

    df['cluster'] = cluster_assignments
    return X, cluster_assignments

def evaluate(X, cluster_assignments):
    silhouette = silhouette_score(X, cluster_assignments)
    db_index = davies_bouldin_score(X, cluster_assignments)

    return silhouette, db_index


# # ### 3.2. Converting categorical features to numerical features

# # In[37]:


# # Create a dictionary to store the mappings of original values to encoded values





# # Display the encoding mappings
# for feature, mapping in encoding_mapping.items():
#     print(f"\nEncoding mapping for {feature}:")
#     for original_value, encoded_value in mapping.items():
#         print(f"{original_value} : {encoded_value}")


# # In[38]:


# df.dtypes


# # In[39]:


# df.head()


# # ## <span style="color:blue;"> 4. Feature Scaling (not recommended after several tests) </span> <a class="anchor" id="feature-scaling"></a>
# # 

# # In[40]:





# # In[24]:


# df.head()


# # ## <span style="color:blue;"> 5. Computing Optimal K for K-Means Clustering </span> <a class="anchor" id="computing-optimal-k-for-clustering"></a>

# # In[40]:




# # plt.figure(figsize=(14, 7))
# # plt.plot(range(1, 10), inertia, marker='o')
# # plt.title('Elbow Method for Optimal k')
# # plt.xlabel('Number of Clusters (k)')
# # plt.ylabel('Explained Variance')
# # plt.show()


# # ## <span style="color:blue;"> 6. Clustering our Dataset </span> <a class="anchor" id="clustering-our-dataset"></a>

# # In[41]:




# df.head()


# # ### Visualizing frequency distribution for each cluster

# # In[42]:


# sns.set_palette("Blues")
# ax = sns.countplot(x='cluster', data=df)
# for p in ax.patches:
#     ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha='center', va='top', xytext=(0, 10), textcoords='offset points')

# plt.title('Cluster Frequency')
# plt.xlabel('Cluster')
# plt.ylabel('Frequency')
# plt.show()


# # ### Visualizing feature distribution by cluster

# # #### Cluster 0

# # In[43]:


# cluster_0_data = df[df['cluster'] == 0]

# # Define the features for which you want to create histograms
# features_to_plot = ['category', 'gender_age_group', 'payment_method', 'shopping_mall', 'total_spending']

# # Create subplots
# fig, axes = plt.subplots(nrows=len(features_to_plot), ncols=1, figsize=(18, 6 * len(features_to_plot)))


# for i, feature in enumerate(features_to_plot):
#     ax = axes[i]
#     sns.histplot(x=feature, data=cluster_0_data, ax=ax, kde=True)
#     ax.set_title(f'Distribution of {feature} in Cluster 0')

# plt.tight_layout()
# plt.show()


# # #### Cluster 1

# # In[44]:


# cluster_0_data = df[df['cluster'] == 1]

# features_to_plot = ['category', 'gender_age_group', 'payment_method', 'shopping_mall', 'total_spending']

# fig, axes = plt.subplots(nrows=len(features_to_plot), ncols=1, figsize=(18, 6 * len(features_to_plot)))
# for i, feature in enumerate(features_to_plot):
#     ax = axes[i]
#     sns.histplot(x=feature, data=cluster_0_data, ax=ax, kde=True)
#     ax.set_title(f'Distribution of {feature} in Cluster 1')

# plt.tight_layout()
# plt.show()


# # #### Cluster 2

# # In[45]:


# cluster_0_data = df[df['cluster'] == 2]

# features_to_plot = ['category', 'gender_age_group', 'payment_method', 'shopping_mall', 'total_spending']


# fig, axes = plt.subplots(nrows=len(features_to_plot), ncols=1, figsize=(18, 6 * len(features_to_plot)))
# for i, feature in enumerate(features_to_plot):
#     ax = axes[i]
#     sns.histplot(x=feature, data=cluster_0_data, ax=ax, kde=True)
#     ax.set_title(f'Distribution of {feature} in Cluster 2')

# plt.tight_layout()
# plt.show()


# # #### Cluster 3

# # In[46]:


# cluster_0_data = df[df['cluster'] == 3]

# features_to_plot = ['category', 'gender_age_group', 'payment_method', 'shopping_mall', 'total_spending']

# fig, axes = plt.subplots(nrows=len(features_to_plot), ncols=1, figsize=(18, 6 * len(features_to_plot)))
# for i, feature in enumerate(features_to_plot):
#     ax = axes[i]
#     sns.histplot(x=feature, data=cluster_0_data, ax=ax, kde=True)
#     ax.set_title(f'Distribution of {feature} in Cluster 3')

# plt.tight_layout()
# plt.show()


# # ### Plotting Cluster scattering

# # In[47]:


# plt.figure(figsize=(16, 6))
# sns.scatterplot(x='gender_age_group', 
#                 y='total_spending', 
#                 data=df, 
#                 hue='cluster', 
#                 palette='viridis',
#                 alpha=0.7, 
#                 legend='full')

# # Adding centroids
# centroids = df.groupby('cluster')[['gender_age_group', 'total_spending']].mean().reset_index()
# sns.scatterplot(x='gender_age_group', 
#                 y='total_spending', 
#                 data=centroids, 
#                 marker='X', 
#                 s=100, 
#                 color='black', 
#                 label='Centroid')

# plt.title('K-means Clustering: Scatter Plot')
# plt.xlabel('Age')
# plt.ylabel('Total Spending')
# plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()


# # ## <span style="color:blue;"> 7. Evaluating our Model's Performance </span> <a class="anchor" id="evaluating-our-model-performance"></a>

# # ### 7.1. Silhouette score (higher is better)

# # In[48]:


# silhouette = silhouette_score(X, df['cluster'])

# silhouette_percentage = silhouette * 100

# print(f"Silhouette Score: {silhouette_percentage:.2f}%")


# # ### 7.2. Davies-Bouldin Index (lower is better)

# # In[49]:


# db_index = davies_bouldin_score(X, df['cluster'])

# db_percentage = db_index * 100

# print(f"Davies-Bouldin Index: {db_percentage:.2f}%")


# # ## <span style="color:blue;"> 8. Conclusion </span> <a class="anchor" id="conclusion"></a>

# # * Conclusion 1;
# # * Conclusion 2;
# # * Conclusion 3;
# # * Conclusion 4.
