import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Path Resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "..", "data", "Urban_Logistics_Node_Path_Dataset.csv")
df = pd.read_csv(file_path)

print("--- LAB 4: FREQUENT PATTERN MINING (APRIORI) ---")

# 1. Convert continuous & categorical route data into boolean transactional format
df_basket = pd.DataFrame()
df_basket['traffic_High'] = df['traffic_density'] == 'High'
df_basket['traffic_Low'] = df['traffic_density'] == 'Low'
df_basket['route_Optimal'] = df['target_route_selection'] == 'Optimal'
df_basket['route_Avoid'] = df['target_route_selection'] == 'Avoid'
df_basket['high_cost'] = df['delivery_cost'] > df['delivery_cost'].median()
df_basket['high_congestion'] = df['congestion_probability'] > df['congestion_probability'].median()

# 2. Apply Apriori Algorithm (Support threshold = 0.10)
frequent_itemsets = apriori(df_basket, min_support=0.10, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets.head())

# 3. Generate Association Rules (Confidence threshold = 0.50)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.50)
print("\nGenerated Association Rules (Support, Confidence, Lift):")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))