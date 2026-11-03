#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("india_housing_prices.csv")

df.head()


# In[3]:


# Check the number of rows and columns
print("Dataset Shape:", df.shape)

# Check column names
print("\nColumn Names:")
print(df.columns.tolist())

# Check data types and missing values
print("\nDataset Information:")
df.info()

# Check missing values in each column
print("\nMissing Values:")
print(df.isnull().sum())


# In[4]:


from sklearn.preprocessing import MinMaxScaler

# Select numerical columns for scaling
numerical_cols = [
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Year_Built",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals"
]

# Create a copy of the original dataset
df_scaled = df.copy()

# Create MinMaxScaler
scaler = MinMaxScaler()

# Scale numerical columns
df_scaled[numerical_cols] = scaler.fit_transform(
    df[numerical_cols]
)

# Display the scaled values
df_scaled[numerical_cols].head()


# In[5]:


# Identify categorical columns
categorical_cols = df.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("Categorical Columns:")
print(categorical_cols)


# In[9]:


print("Categorical Encoding Started")

print("Original shape:", df.shape)
print("Categorical columns:", len(categorical_cols))


# In[11]:


import pandas as pd

# Encode categorical features
df_encoded = pd.get_dummies(
    df,
    columns=["State", "City", "Locality", "Property_Type"],
    dtype=int
)

# Display the first 5 rows
df_encoded.head()


# In[15]:


# Create new features

df["Nearby_Facilities_Total"] = (
    df["Nearby_Schools"] + df["Nearby_Hospitals"]
)

df["School_Density_Score"] = df["Nearby_Schools"]

# Display the new features
df[[
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Nearby_Facilities_Total",
    "School_Density_Score"
]].head()


# In[16]:


print(df[[
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Nearby_Facilities_Total",
    "School_Density_Score"
]].head().to_string(index=False))


# In[17]:


df["Price_per_SqFt"].median()


# In[18]:


df["Price_per_SqFt"].describe()


# In[19]:


print(df[["Price_in_Lakhs", "Price_per_SqFt"]].head())


# In[20]:


# Create a proxy label for Good Investment
good_price = df["Price_per_SqFt"] <= df["Price_per_SqFt"].median()
good_schools = df["Nearby_Schools"] >= df["Nearby_Schools"].median()
good_hospitals = df["Nearby_Hospitals"] >= df["Nearby_Hospitals"].median()

df["Good_Investment"] = (
    good_price & good_schools & good_hospitals
).astype(int)

# Check label counts
print(df["Good_Investment"].value_counts())


# In[22]:


df[[
    "Price_per_SqFt",
    "Nearby_Facilities_Total",
    "School_Density_Score",
    "Good_Investment"
]].head().to_string(index=False)


# In[23]:


# Separate features (X) and target (y)

X = df.drop(columns=[
    "ID",
    "Good_Investment",
    "Price_per_SqFt",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Nearby_Facilities_Total",
    "School_Density_Score"
])

y = df["Good_Investment"]

print("X shape:", X.shape)
print("y shape:", y.shape)


# In[24]:


df["Public_Transport_Accessibility"].value_counts()


# In[25]:


# Define investment conditions
good_price = df["Price_per_SqFt"] <= df["Price_per_SqFt"].median()

good_schools = df["Nearby_Schools"] >= df["Nearby_Schools"].median()

good_hospitals = df["Nearby_Hospitals"] >= df["Nearby_Hospitals"].median()

good_transport = df["Public_Transport_Accessibility"] == "High"

# Create updated Good_Investment label
df["Good_Investment"] = (
    good_price &
    good_schools &
    good_hospitals &
    good_transport
).astype(int)

# Check the result
print(df["Good_Investment"].value_counts())


# In[26]:


# Separate features (X) and target (y)

X = df.drop(columns=[
    "ID",
    "Good_Investment",
    "Price_per_SqFt",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Nearby_Facilities_Total",
    "School_Density_Score"
])

y = df["Good_Investment"]

print("X shape:", X.shape)
print("y shape:", y.shape)


# In[27]:


categorical_cols = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print(categorical_cols)


# In[28]:


for col in categorical_cols:
    print(col, ":", X[col].nunique())


# In[29]:


# Select categorical columns except Locality and Amenities

small_cat_cols = [
    col for col in categorical_cols
    if col not in ["Locality", "Amenities"]
]

# One-Hot Encoding
X_encoded = pd.get_dummies(
    X,
    columns=small_cat_cols,
    dtype=int
)

print("Original X shape:", X.shape)
print("Encoded X shape:", X_encoded.shape)


# In[30]:


# Frequency Encoding for Locality

locality_freq = X["Locality"].value_counts(normalize=True)

X_encoded["Locality_Frequency"] = (
    X["Locality"].map(locality_freq)
)

X_encoded.head()


# In[31]:


# Remove original Locality column
X_encoded = X_encoded.drop(columns=["Locality"])

# Check the result
X_encoded[["Locality_Frequency"]].head()


# In[32]:


X["Amenities"].head(10)


# In[33]:


# Split amenities into separate binary columns
amenities_encoded = X["Amenities"].str.get_dummies(sep=", ")

# Add encoded columns to X_encoded
X_encoded = pd.concat(
    [X_encoded, amenities_encoded],
    axis=1
)

# Display the result
amenities_encoded.head()


# In[36]:


print(X_encoded.columns.tolist())


# In[37]:


X_encoded.select_dtypes(include=["object", "str"]).columns.tolist()


# In[40]:


[col for col in X_encoded.columns if "Public_Transport" in col]


# In[41]:


X_encoded.shape


# In[42]:


city_price = df.groupby("City")["Price_in_Lakhs"].mean()

city_price = city_price.sort_values(ascending=False)

city_price.head(10)


# In[44]:


ax = city_price.head(10).sort_values().plot(
    kind="barh",
    figsize=(10, 6)
)

ax.bar_label(ax.containers[0], fmt="%.2f")

plt.title("Top 10 Cities by Average Property Price")
plt.xlabel("Average Price (Lakhs)")
plt.ylabel("City")
plt.show()


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[9]:


print(df.columns.tolist())


# In[10]:


sns.scatterplot(
    data=df.sample(min(5000, len(df)), random_state=1),
    x="Size_in_SqFt",
    y="Price_in_Lakhs",
    alpha=0.3
)

plt.title("Property Area vs Price")
plt.xlabel("Area (SqFt)")
plt.ylabel("Price (Lakhs)")
plt.show()


# In[11]:


df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])


# In[12]:


df["Security"].value_counts()


# In[13]:


sns.countplot(data=df, x="Security")

plt.title("Property Security Availability")
plt.xlabel("Security")
plt.ylabel("Number of Properties")
plt.show()


# In[14]:


sns.boxplot(
    data=df,
    x="Security",
    y="Price_in_Lakhs"
)

plt.title("Security vs Property Price")
plt.xlabel("Security Availability")
plt.ylabel("Property Price (Lakhs)")
plt.show()


# In[15]:


df.groupby("Security")["Price_in_Lakhs"].mean()


# In[19]:


# Convert accessibility into numeric scores
df["Transport_Score"] = df["Public_Transport_Accessibility"].map({
    "High": 3,
    "Medium": 2,
    "Low": 1
})

# Create infrastructure score
df["Infrastructure_Score"] = (
    df["Transport_Score"] +
    df["Nearby_Schools"] +
    df["Nearby_Hospitals"]
)


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    data=df.sample(5000, random_state=1),
    x="Infrastructure_Score",
    y="Price_in_Lakhs",
    alpha=0.3
)

plt.title("Infrastructure Score vs Property Price")
plt.xlabel("Infrastructure Score")
plt.ylabel("Property Price (Lakhs)")
plt.show()


# In[21]:


df["Infrastructure_Score"].corr(df["Price_in_Lakhs"])


# What is the distribution of property prices?

# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(data=df, x="Price_in_Lakhs", bins=30, kde=True)

plt.title("Distribution of Property Prices")
plt.xlabel("Price (Lakhs)")
plt.ylabel("Frequency")
plt.show()


# What is the distribution of property sizes?

# In[7]:


plt.figure(figsize=(8,5))
sns.histplot(data=df, x="Size_in_SqFt", bins=30, kde=True)

plt.title("Distribution of Property Sizes")
plt.xlabel("Size (SqFt)")
plt.ylabel("Frequency")
plt.show()


# How does price per sq ft vary by property type?

# In[24]:


df.groupby("Property_Type")["Price_per_SqFt"].mean()


# In[25]:


plt.figure(figsize=(9,5))
sns.barplot(data=df, x="Property_Type", y="Price_per_SqFt")

plt.title("Average Price per SqFt by Property Type")
plt.xlabel("Property Type")
plt.ylabel("Price per SqFt")
plt.xticks(rotation=45)
plt.show()


# Is there a relationship between property size and price?

# In[26]:


sample_df = df.sample(5000, random_state=1)

sns.scatterplot(
    data=sample_df,
    x="Size_in_SqFt",
    y="Price_in_Lakhs",
    alpha=0.3
)

plt.title("Property Size vs Property Price")
plt.xlabel("Size (SqFt)")
plt.ylabel("Price (Lakhs)")
plt.show()


# In[27]:


df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])


# Are there any outliers in price per sq ft or property size?

# In[29]:


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.boxplot(y=df["Price_per_SqFt"])
plt.title("Outliers in Price per SqFt")

plt.subplot(1,2,2)
sns.boxplot(y=df["Size_in_SqFt"])
plt.title("Outliers in Property Size")

plt.tight_layout()
plt.show()


# What is the average price per sq ft by state?

# In[30]:


df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False)


# In[31]:


plt.figure(figsize=(12,5))

sns.barplot(
    data=df,
    x="State",
    y="Price_per_SqFt",
    estimator="mean"
)

plt.title("Average Price per SqFt by State")
plt.xticks(rotation=45)
plt.show()


# What is the average property price by city?

# In[32]:


df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False)


# In[33]:


plt.figure(figsize=(12,5))

sns.barplot(
    data=df,
    x="City",
    y="Price_in_Lakhs",
    estimator="mean"
)

plt.title("Average Property Price by City")
plt.xticks(rotation=90)
plt.show()


# What is the median age of properties by locality?

# In[34]:


top_localities = (
    df.groupby("Locality")["Age_of_Property"]
    .median()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)

sns.barplot(
    data=top_localities,
    x="Locality",
    y="Age_of_Property"
)

plt.title("Median Property Age by Locality")
plt.xticks(rotation=90)
plt.show()


# How is BHK distributed across cities?

# In[35]:


bhk_distribution = pd.crosstab(df["City"], df["BHK"])

bhk_distribution


# In[36]:


bhk_distribution.plot(
    kind="bar",
    stacked=True,
    figsize=(12,6)
)

plt.title("BHK Distribution Across Cities")
plt.xlabel("City")
plt.ylabel("Number of Properties")
plt.xticks(rotation=90)
plt.show()


# What are the price trends for the top 5 most expensive localities?

# In[37]:


top_5_localities = (
    df.groupby("Locality")["Price_in_Lakhs"]
    .mean()
    .nlargest(5)
)

top_5_localities


# In[38]:


top_5_df = df[df["Locality"].isin(top_5_localities.index)]

plt.figure(figsize=(10,5))

sns.barplot(
    data=top_5_df,
    x="Locality",
    y="Price_in_Lakhs",
    estimator="mean"
)

plt.title("Top 5 Most Expensive Localities")
plt.xticks(rotation=45)
plt.show()


# How are numeric features correlated with each other?

# In[39]:


numeric_df = df.select_dtypes(include="number")

correlation = numeric_df.corr()

correlation


# How are numeric features correlated with each other?

# In[41]:


numeric_df = df.select_dtypes(include="number")

numeric_df = numeric_df.drop(
    columns=["ID"],
    errors="ignore"
)

plt.figure(figsize=(14, 9))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()


# How do nearby schools relate to price per sq ft?

# In[42]:


df["Nearby_Schools"].corr(df["Price_per_SqFt"])


# In[43]:


df.groupby("Nearby_Schools")["Price_per_SqFt"].mean()


# In[44]:


plt.figure(figsize=(9,5))

sns.barplot(
    data=df,
    x="Nearby_Schools",
    y="Price_per_SqFt"
)

plt.title("Nearby Schools vs Price per SqFt")
plt.xlabel("Number of Nearby Schools")
plt.ylabel("Average Price per SqFt")
plt.show()


# How do nearby hospitals relate to price per sq ft?

# In[46]:


df["Nearby_Hospitals"].corr(df["Price_per_SqFt"])


# How does price vary by furnished status?

# In[47]:


df.groupby("Furnished_Status")["Price_in_Lakhs"].mean()


# In[9]:


plt.figure(figsize=(8,5))

sns.barplot(
    data=df,
    x="Furnished_Status",
    y="Price_in_Lakhs",
    estimator="mean"
)

plt.title("Average Property Price by Furnished Status")
plt.xlabel("Furnished Status")
plt.ylabel("Average Price (Lakhs)")
plt.show()


# How does price per sq ft vary by property facing direction?

# In[49]:


df.groupby("Facing")["Price_per_SqFt"].mean().sort_values(ascending=False)


# In[50]:


plt.figure(figsize=(8,5))

sns.barplot(
    data=df,
    x="Facing",
    y="Price_per_SqFt",
    estimator="mean"
)

plt.title("Average Price per SqFt by Facing Direction")
plt.xlabel("Facing Direction")
plt.ylabel("Average Price per SqFt")
plt.show()


# How many properties belong to each owner type?

# In[2]:


df["Owner_Type"].value_counts()


# In[10]:


ax = sns.countplot(data=df, x="Owner_Type")

ax.bar_label(ax.containers[0], padding=3)

plt.title("Number of Properties by Owner Type")
plt.xlabel("Owner Type")
plt.ylabel("Number of Properties")
plt.xticks(rotation=45)
plt.show()


# How many properties are available under each availability status?

# In[11]:


df["Availability_Status"].value_counts()


# In[12]:


bx = sns.countplot(data=df, x="Availability_Status")
bx.bar_label(bx.containers[0], padding=3)
plt.title("Properties by Availability Status")
plt.xlabel("Availability Status")
plt.ylabel("Number of Properties")
plt.xticks(rotation=45)
plt.show()


# Does parking space affect property price?

# In[13]:


df.groupby("Parking_Space")["Price_in_Lakhs"].mean()


# In[14]:


cx = sns.barplot(
    data=df,
    x="Parking_Space",
    y="Price_in_Lakhs",
    estimator="mean"
)
cx.bar_label(cx.containers[0], padding=3)

plt.title("Parking Space vs Property Price")
plt.xlabel("Parking Space")
plt.ylabel("Average Price (Lakhs)")
plt.show()


# How do amenities affect price per sq ft?

# In[15]:


df.groupby("Amenities")["Price_per_SqFt"].mean().sort_values(ascending=False)


# In[17]:


df["Amenities"].nunique()


# In[18]:


df["Amenities"].value_counts().head(10)


# In[19]:


top_amenities = df["Amenities"].value_counts().head(10).index

amenity_price = (
    df[df["Amenities"].isin(top_amenities)]
    .groupby("Amenities")["Price_per_SqFt"]
    .mean()
    .sort_values(ascending=True)
    .reset_index()
)

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=amenity_price,
    x="Price_per_SqFt",
    y="Amenities"
)

ax.bar_label(ax.containers[0], fmt="%.3f", padding=3)

plt.title("Top 10 Amenities vs Average Price per SqFt")
plt.xlabel("Average Price per SqFt")
plt.ylabel("Amenities")
plt.tight_layout()
plt.show()


# How does public transport accessibility relate to price per sq ft or investment potential?

# In[20]:


df.groupby("Public_Transport_Accessibility")["Price_per_SqFt"].mean()


# In[22]:


px = sns.barplot(
    data=df,
    x="Public_Transport_Accessibility",
    y="Price_per_SqFt",
    estimator="mean"
)
px.bar_label(px.containers[0], padding=3)

plt.title("Transport Accessibility vs Price per SqFt")
plt.xlabel("Transport Accessibility")
plt.ylabel("Average Price per SqFt")
plt.show()


# In[3]:


df["City"].value_counts().head(10)


# In[3]:


df.columns


# classification

# In[4]:


median_price = df["Price_in_Lakhs"].median()

print("Median Property Price:", median_price)


# In[5]:


median_price_sqft = df["Price_per_SqFt"].median()

print("Median Price per SqFt:", median_price_sqft)


# Create Good_Investment Column
# 

# In[7]:


df["Good_Investment"] = (
    (df["Price_in_Lakhs"] <= median_price) &
    (df["Price_per_SqFt"] <= median_price_sqft)
).astype(int)


# In[8]:


df["Good_Investment"].value_counts()


# In[9]:


df[["Price_in_Lakhs", "Price_per_SqFt", "Good_Investment"]].head(10)


# In[10]:


features = [
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Year_Built",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Floor_No",
    "Total_Floors"
]

X = df[features]
y = df["Good_Investment"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)


# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# In[12]:
# Random Forest Model with MLflow Tracking

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

mlflow.set_experiment("Real Estate Investment Prediction")

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Model Training Completed!")
    print("Model Accuracy:", accuracy * 100, "%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

       # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(
        model,
        name="random_forest_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print("MLflow Logging Completed!")

# In[14]:


import pandas as pd

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)


# In[15]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Random Forest - Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[16]:


y_pred_proba = model.predict_proba(X_test)

print(y_pred_proba[:5])


# In[17]:


confidence = y_pred_proba.max(axis=1)

print(confidence[:10])


# Future Price Calculation

# In[19]:


growth_rate = 0.08
years = 5

future_price = df["Price_in_Lakhs"] * (1 + growth_rate) ** years

df["Estimated_Price_After_5_Years"] = future_price

df[["Price_in_Lakhs", "Estimated_Price_After_5_Years"]].head(10)


# In[20]:


df["Expected_Price_Increase"] = (
    df["Estimated_Price_After_5_Years"] - df["Price_in_Lakhs"]
)

df[[
    "Price_in_Lakhs",
    "Estimated_Price_After_5_Years",
    "Expected_Price_Increase"
]].head(10)


# Calculate Average Future Price

# In[21]:


# Average Current Price vs Future Price

labels = [
    "Current Price",
    "Future Price",
    "Expected Increase"
]

values = [
    df["Price_in_Lakhs"].mean(),
    df["Estimated_Price_After_5_Years"].mean(),
    df["Expected_Price_Increase"].mean()
]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values)

plt.title("Average Current Price vs Future Price")
plt.ylabel("Price in Lakhs")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"₹{bar.get_height():.2f}",
        ha="center",
        va="bottom"
    )

plt.show()


# City-wise Future Price Estimation

# In[24]:


city_analysis = df.groupby("City").agg(
    Average_Current_Price=("Price_in_Lakhs", "mean"),
    Average_Future_Price=("Estimated_Price_After_5_Years", "mean")
).reset_index()

city_analysis.head(10)


# In[25]:


city_top10 = city_analysis.head(10)

city_top10.plot(
    x="City",
    y=["Average_Current_Price", "Average_Future_Price"],
    kind="bar",
    figsize=(12, 6)
)

plt.title("City-wise Current vs Future Property Price")
plt.xlabel("City")
plt.ylabel("Price in Lakhs")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


#  Property Type-wise Future Price Analysis

# In[26]:


property_analysis = df.groupby("Property_Type").agg(
    Average_Current_Price=("Price_in_Lakhs", "mean"),
    Average_Future_Price=("Estimated_Price_After_5_Years", "mean")
).reset_index()

property_analysis


# In[27]:


property_analysis.plot(
    x="Property_Type",
    y=["Average_Current_Price", "Average_Future_Price"],
    kind="bar",
    figsize=(10, 5)
)

plt.title("Property Type-wise Current vs Future Price")
plt.xlabel("Property Type")
plt.ylabel("Price in Lakhs")
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.show()


# In[28]:


print("Total Properties:", len(df))

print("Average Current Price:",
      round(df["Price_in_Lakhs"].mean(), 2))

print("Average Future Price:",
      round(df["Estimated_Price_After_5_Years"].mean(), 2))

print("Average Expected Increase:",
      round(df["Expected_Price_Increase"].mean(), 2))

print("Annual Growth Rate: 8%")
print("Prediction Period: 5 Years")


# In[29]:


df.to_csv("real_estate_processed.csv", index=False)


# In[30]:


import joblib

joblib.dump(model, "investment_model.pkl")


# In[ ]:




