import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Credit Risk Dashboard",
    layout="wide"
)

st.title("📊 Credit Risk Analysis Dashboard")

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    credit = pd.read_csv("credit_record.csv")
    app = pd.read_csv("application_record.csv")

    df = pd.merge(credit, app, on="ID")

    df.rename(columns={
        'AMT_INCOME_TOTAL': 'Annual_Income',
        'CNT_FAM_MEMBERS': 'Family_Member_Count',
        'DAYS_BIRTH': 'Age_Days',
        'DAYS_EMPLOYED': 'Employment_Days',
        'NAME_INCOME_TYPE': 'Income_Type',
        'OCCUPATION_TYPE': 'Occupation_Type'
    }, inplace=True)

    df['Occupation_Type'] = df['Occupation_Type'].fillna('Laborers')

    df['Credit_Risk'] = df['STATUS']
    df['Credit_Risk'] = df['Credit_Risk'].replace(['2', '3', '4', '5'], 1)
    df['Credit_Risk'] = df['Credit_Risk'].replace(['1', '0', 'C', 'X'], 0)

    df['Age_Days'] = abs(df['Age_Days']) // 365
    df['Employment_Days'] = abs(df['Employment_Days']) // 365

    return df

df = load_data()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Filters")

income_filter = st.sidebar.slider(
    "Annual Income",
    int(df['Annual_Income'].min()),
    int(df['Annual_Income'].max()),
    (
        int(df['Annual_Income'].min()),
        int(df['Annual_Income'].max())
    )
)

filtered_df = df[
    (df['Annual_Income'] >= income_filter[0]) &
    (df['Annual_Income'] <= income_filter[1])
]

# ---------------------------
# Dataset Preview
# ---------------------------
st.subheader("Dataset Preview")
st.dataframe(filtered_df.head())

# ---------------------------
# KPI Cards
# ---------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(filtered_df))

with col2:
    st.metric("Average Income",
              f"{filtered_df['Annual_Income'].mean():,.0f}")

with col3:
    st.metric("Average Age",
              f"{filtered_df['Age_Days'].mean():.0f}")

# ---------------------------
# Credit Risk Distribution
# ---------------------------
st.subheader("Credit Risk Distribution")

fig, ax = plt.subplots()
sns.countplot(x='Credit_Risk', data=filtered_df, ax=ax)
st.pyplot(fig)

# ---------------------------
# Annual Income Distribution
# ---------------------------
st.subheader("Annual Income Distribution")

fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(filtered_df['Annual_Income'], bins=20, ax=ax)
st.pyplot(fig)

# ---------------------------
# Age Distribution
# ---------------------------
st.subheader("Age Distribution")

fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(filtered_df['Age_Days'], bins=15, ax=ax)
st.pyplot(fig)

# ---------------------------
# Employment Years
# ---------------------------
st.subheader("Employment Length Distribution")

fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(filtered_df['Employment_Days'], bins=10, ax=ax)
st.pyplot(fig)

# ---------------------------
# Occupation vs Credit Risk
# ---------------------------
st.subheader("Occupation Type vs Credit Risk")

fig, ax = plt.subplots(figsize=(12,5))
sns.barplot(
    x='Occupation_Type',
    y='Credit_Risk',
    data=filtered_df,
    ax=ax
)
plt.xticks(rotation=90)
st.pyplot(fig)

# ---------------------------
# Data Preparation
# ---------------------------
model_df = df.copy()

model_df = model_df.drop([
    'ID', 'STATUS', 'CODE_GENDER',
    'CNT_CHILDREN', 'Age_Days',
    'FLAG_MOBIL', 'FLAG_WORK_PHONE',
    'FLAG_PHONE', 'FLAG_EMAIL'
], axis=1)

label = LabelEncoder()

cols = [
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'Income_Type',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'Occupation_Type'
]

for col in cols:
    model_df[col] = label.fit_transform(model_df[col])

X = model_df.drop("Credit_Risk", axis=1)
y = model_df["Credit_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ---------------------------
# Model Training
# ---------------------------
st.subheader("Model Performance")

model_option = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Random Forest", "XGBoost"]
)

if st.button("Train Model"):

    if model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif model_option == "Random Forest":
        model = RandomForestClassifier()

    else:
        model = XGBClassifier()

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    st.success(
        f"{model_option} Accuracy: {acc*100:.2f}%"
    )

# ---------------------------
# Correlation Heatmap
# ---------------------------
st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(
    model_df.corr(numeric_only=True),
    cmap="coolwarm",
    ax=ax
)
st.pyplot(fig)

st.markdown("---")
st.write("Credit Risk Analysis Dashboard using Streamlit")
