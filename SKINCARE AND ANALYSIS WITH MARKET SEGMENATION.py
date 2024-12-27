#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[2]:


# Step 1: Generate Random Data
np.random.seed(42)  # for reproducibility

# Creating a sample dataset with 1000 users
n = 1000
data = pd.DataFrame({
    'user_id': range(1, n+1),
    'age': np.random.randint(18, 60, size=n),  # ages between 18 and 60
    'skin_type': np.random.choice(['Oily', 'Dry', 'Combination', 'Sensitive'], size=n),
    'acne_severity': np.random.randint(1, 5, size=n),  # 1: Low, 4: High
    'dark_spots': np.random.choice([0, 1], size=n),  # 0: No, 1: Yes
    'recommended_product': np.random.choice(['Product_A', 'Product_B', 'Product_C', 'Product_D'], size=n),
    'skin_improvement_score': np.random.normal(5, 2, n).clip(0, 10)  # normalized score 0-10
})


# In[3]:


# Step 2: Perform EDA

# Summary of the data
print("Data Summary:")
print(data.describe(include='all'))


# In[4]:


# Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], kde=True, bins=20, color='skyblue')
plt.title('Age Distribution of Users')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[5]:


# Skin Type Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='skin_type', palette='Set2')
plt.title('Distribution of Skin Types')
plt.xlabel('Skin Type')
plt.ylabel('Count')
plt.show()


# In[6]:


# Acne Severity by Skin Type
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='skin_type', y='acne_severity', palette='Set3')
plt.title('Acne Severity by Skin Type')
plt.xlabel('Skin Type')
plt.ylabel('Acne Severity')
plt.show()


# In[7]:


# Correlation between Age and Skin Improvement Score
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='age', y='skin_improvement_score', hue='skin_type', palette='viridis')
plt.title('Age vs. Skin Improvement Score')
plt.xlabel('Age')
plt.ylabel('Skin Improvement Score')
plt.show()


# In[8]:


# Distribution of Skin Improvement Scores by Recommended Product
plt.figure(figsize=(12, 6))
sns.violinplot(data=data, x='recommended_product', y='skin_improvement_score', palette='muted')
plt.title('Skin Improvement Scores by Recommended Product')
plt.xlabel('Recommended Product')
plt.ylabel('Skin Improvement Score')
plt.show()


# In[9]:


# Heatmap of Correlations
plt.figure(figsize=(8, 5))
# Select only numerical columns for correlation calculation
numerical_data = data.select_dtypes(include=np.number) 
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[ ]:





# In[10]:


# Step 3: Encoding Categorical Variables
label_encoder = LabelEncoder()
data['skin_type_encoded'] = label_encoder.fit_transform(data['skin_type'])
data['recommended_product_encoded'] = label_encoder.fit_transform(data['recommended_product'])


# In[11]:


# Step 4: Machine Learning Model
# Define input features and target variable
X = data[['age', 'skin_type_encoded', 'acne_severity', 'dark_spots']]
y = data['recommended_product_encoded']


# In[12]:


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[13]:


# Evaluate the model
predictions = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, predictions))


# In[14]:


# Step 5: Market Segmentation using K-Means
# Prepare features for clustering
features = data[['age', 'acne_severity', 'skin_improvement_score']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# In[15]:


# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
data['segment'] = kmeans.fit_predict(scaled_features)


# In[16]:


# Visualize market segments
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['age'], y=data['skin_improvement_score'], hue=data['segment'], palette='viridis')
plt.title('Market Segments')
plt.xlabel('Age')
plt.ylabel('Skin Improvement Score')
plt.show()


# In[17]:


# Step 6: Financial Model (Example Calculation)
# Define parameters
subscription_fee = 10  # Monthly subscription fee in dollars
active_users = len(data)  # Number of users
affiliate_commission = 5  # Commission per product sold
units_sold = 500  # Estimated product sales
fixed_costs = 5000  # Monthly fixed costs


# In[18]:


# Revenue and Profit Calculation
revenue = (subscription_fee * active_users) + (affiliate_commission * units_sold)
profit = revenue - fixed_costs

print(f"Total Revenue: ${revenue}")
print(f"Total Profit: ${profit}")


# In[ ]:





# In[ ]:




