```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Prediction Dashboard",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}

[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #e6e6e6;
    padding: 15px;
    border-radius: 10px;
}

h1, h2, h3 {
    color: #0E4D92;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🏦 Credit Risk Prediction Dashboard")
st.markdown("### Customer Credit Risk Analysis using Machine Learning")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
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
    df['Credit_Risk'] = df['Credit_Risk'].replace(['2','3','4','5'], 1)
    df['Credit_Risk'] = df['Credit_Risk'].replace(['1','0','C','X'], 0)

    df['Age_Days'] = abs(df['Age_Days']) // 365
    df['Employment_Days'] = abs(df['Employment_Days']) // 365

    return df

try:
    df = load_data()

except Exception as e:
    st.error(
        "Dataset files not found.\n\n"
        "Make sure credit_record.csv and application_record.csv "
        "are in the same folder as app.py."
    )
    st.stop()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("📂 Dashboard Filters")

income_range = st.sidebar.slider(
    "Select Annual Income Range",
    int(df["Annual_Income"].min()),
    int(df["Annual_Income"].max()),
    (
        int(df["Annual_Income"].min()),
        int(df["Annual_Income"].max())
    )
)

filtered_df = df[
    (df["Annual_Income"] >= income_range[0]) &
    (df["Annual_Income"] <= income_range[1])
]

# --------------------------------------------------
# KPI Section
# --------------------------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        f"{len(filtered_df):,}"
    )

with col2:
    st.metric(
        "Average Income",
        f"₹{filtered_df['Annual_Income'].mean():,.0f}"
    )

with col3:
    st.metric(
        "Average Age",
        f"{filtered_df['Age_Days'].mean():.0f} Years"
    )

with col4:
    risk_rate = filtered_df["Credit_Risk"].mean() * 100
    st.metric(
        "Risk Rate",
        f"{risk_rate:.2f}%"
    )

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📋 Overview",
    "📈 Visualizations",
    "🤖 Machine Learning"
])

# --------------------------------------------------
# TAB 1
# --------------------------------------------------
with tab1:

    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(10))

    st.subheader("Credit Risk Distribution")

    fig, ax = plt.subplots(figsize=(8,4))
    sns.countplot(
        x="Credit_Risk",
        data=filtered_df,
        ax=ax
    )

    ax.set_title("Credit Risk Count")
    st.pyplot(fig)

    st.subheader("💡 Business Insights")

    st.info(f"""
    • Total Customers Analysed: {len(filtered_df):,}

    • Credit Risk Percentage: {risk_rate:.2f}%

    • Higher income customers tend to have lower risk.

    • Employment duration can influence repayment behaviour.

    • Occupation type impacts creditworthiness.
    """)

# --------------------------------------------------
# TAB 2
# --------------------------------------------------
with tab2:

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Annual Income Distribution")

        fig, ax = plt.subplots()

        sns.histplot(
            filtered_df["Annual_Income"],
            bins=20,
            kde=True,
            ax=ax
        )

        st.pyplot(fig)

    with col2:

        st.subheader("Age Distribution")

        fig, ax = plt.subplots()

        sns.histplot(
            filtered_df["Age_Days"],
            bins=15,
            kde=True,
            ax=ax
        )

        st.pyplot(fig)

    st.subheader("Employment Length Distribution")

    fig, ax = plt.subplots(figsize=(8,4))

    sns.histplot(
        filtered_df["Employment_Days"],
        bins=10,
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Occupation vs Credit Risk")

    fig, ax = plt.subplots(figsize=(12,5))

    sns.barplot(
        x="Occupation_Type",
        y="Credit_Risk",
        data=filtered_df,
        ax=ax
    )

    plt.xticks(rotation=90)
    st.pyplot(fig)

# --------------------------------------------------
# Data Preparation
# --------------------------------------------------
model_df = df.copy()

model_df = model_df.drop([
    'ID',
    'STATUS',
    'CODE_GENDER',
    'CNT_CHILDREN',
    'Age_Days',
    'FLAG_MOBIL',
    'FLAG_WORK_PHONE',
    'FLAG_PHONE',
    'FLAG_EMAIL'
], axis=1)

encoder = LabelEncoder()

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
    model_df[col] = encoder.fit_transform(
        model_df[col]
    )

X = model_df.drop("Credit_Risk", axis=1)
y = model_df["Credit_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# TAB 3
# --------------------------------------------------
with tab3:

    st.subheader("Machine Learning Model Comparison")

    if st.button("🚀 Train Models"):

        models = {
            "Logistic Regression":
            LogisticRegression(max_iter=1000),

            "Random Forest":
            RandomForestClassifier(),

            "XGBoost":
            XGBClassifier()
        }

        results = {}

        for name, model in models.items():

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            acc = accuracy_score(
                y_test,
                pred
            )

            results[name] = round(acc * 100, 2)

        result_df = pd.DataFrame(
            results.items(),
            columns=[
                "Model",
                "Accuracy (%)"
            ]
        )

        st.dataframe(
            result_df,
            use_container_width=True
        )

        best_model = result_df.sort_values(
            by="Accuracy (%)",
            ascending=False
        ).iloc[0]

        st.success(
            f"🏆 Best Model: "
            f"{best_model['Model']} "
            f"({best_model['Accuracy (%)']}%)"
        )

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12,6))

    sns.heatmap(
        model_df.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "Developed using Streamlit, Scikit-Learn, and XGBoost 🚀"
)
```
