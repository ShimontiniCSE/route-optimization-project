import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

st.set_page_config(page_title="Route Optimization Data Mining UI", layout="wide")
st.title("🚚 Route Optimization — Data Mining Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("data/Urban_Logistics_Node_Path_Dataset.csv")

df = load_data()

option = st.sidebar.selectbox("Select Lab Module", [
    "Lab 1 & 2: Data & Preprocessing", 
    "Lab 3: EDA & Visualization", 
    "Lab 4: Apriori Rules",
    "Lab 5: FP-Growth vs Apriori"
])

if option == "Lab 1 & 2: Data & Preprocessing":
    st.header("Lab 1 & 2: Dataset Overview & Summary")
    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10))
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())

elif option == "Lab 3: EDA & Visualization":
    st.header("Lab 3: Exploratory Data Analysis")
    st.subheader("Path Length vs. Delivery Cost")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df, x="travel_time", y="delivery_cost", hue="target_route_selection", ax=ax)
    st.pyplot(fig)
    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.heatmap(df.select_dtypes(include=["float64", "int64"]).corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

elif option == "Lab 4: Apriori Rules":
    st.header("Lab 4: Association Rule Mining (Apriori)")
    min_sup = st.slider("Minimum Support Threshold", 0.05, 0.50, 0.10)
    min_conf = st.slider("Minimum Confidence Threshold", 0.10, 1.00, 0.50)
    
    df_basket = pd.DataFrame()
    df_basket["traffic_High"] = df["traffic_density"] == "High"
    df_basket["traffic_Low"] = df["traffic_density"] == "Low"
    df_basket["route_Optimal"] = df["target_route_selection"] == "Optimal"
    df_basket["route_Avoid"] = df["target_route_selection"] == "Avoid"
    df_basket["high_cost"] = df["delivery_cost"] > df["delivery_cost"].median()
    df_basket["high_congestion"] = df["congestion_probability"] > df["congestion_probability"].median()
    
    freq_items = apriori(df_basket, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
    st.subheader("Filtered Association Rules")
    st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

elif option == "Lab 5: FP-Growth vs Apriori":
    st.header("Lab 5: FP-Growth Algorithm & Algorithm Comparison")
    
    min_sup = st.slider("Support Threshold", 0.05, 0.50, 0.10, key="fp_sup")
    min_conf = st.slider("Confidence Threshold", 0.10, 1.00, 0.50, key="fp_conf")
    
    df_basket = pd.DataFrame()
    df_basket["traffic_High"] = df["traffic_density"] == "High"
    df_basket["traffic_Low"] = df["traffic_density"] == "Low"
    df_basket["route_Optimal"] = df["target_route_selection"] == "Optimal"
    df_basket["route_Avoid"] = df["target_route_selection"] == "Avoid"
    df_basket["high_cost"] = df["delivery_cost"] > df["delivery_cost"].median()
    df_basket["high_congestion"] = df["congestion_probability"] > df["congestion_probability"].median()
    
    # Benchmarking Execution Time
    t0 = time.time()
    freq_apriori = apriori(df_basket, min_support=min_sup, use_colnames=True)
    t_apriori = time.time() - t0
    
    t1 = time.time()
    freq_fpgrowth = fpgrowth(df_basket, min_support=min_sup, use_colnames=True)
    t_fpgrowth = time.time() - t1
    
    # Layout Fix: Use st.columns instead of st.subplots
    col1, col2 = st.columns(2)
    col1.metric("Apriori Execution Time", f"{t_apriori:.5f} sec")
    col2.metric("FP-Growth Execution Time", f"{t_fpgrowth:.5f} sec")
    
    rules_fp = association_rules(freq_fpgrowth, metric="confidence", min_threshold=min_conf)
    st.subheader("FP-Growth Mining Results")
    st.dataframe(rules_fp[["antecedents", "consequents", "support", "confidence", "lift"]])