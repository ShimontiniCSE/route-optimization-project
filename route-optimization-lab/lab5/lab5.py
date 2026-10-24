import os
import time
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# Path Resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "..", "data", "Urban_Logistics_Node_Path_Dataset.csv")
df = pd.read_csv(file_path)

print("--- LAB 5: FREQUENT PATTERN MINING (FP-GROWTH VS APRIORI) ---")

# 1. Prepare Transactional Data Matrix
df_basket = pd.DataFrame()
df_basket['traffic_High'] = df['traffic_density'] == 'High'
df_basket['traffic_Low'] = df['traffic_density'] == 'Low'
df_basket['route_Optimal'] = df['target_route_selection'] == 'Optimal'
df_basket['route_Avoid'] = df['target_route_selection'] == 'Avoid'
df_basket['high_cost'] = df['delivery_cost'] > df['delivery_cost'].median()
df_basket['high_congestion'] = df['congestion_probability'] > df['congestion_probability'].median()

# 2. Benchmark Apriori Execution Time
start_ap = time.time()
freq_apriori = apriori(df_basket, min_support=0.10, use_colnames=True)
apriori_time = time.time() - start_ap

# 3. Benchmark FP-Growth Execution Time
start_fp = time.time()
freq_fpgrowth = fpgrowth(df_basket, min_support=0.10, use_colnames=True)
fpgrowth_time = time.time() - start_fp

print(f"Apriori Mining Time:   {apriori_time:.6f} seconds")
print(f"FP-Growth Mining Time: {fpgrowth_time:.6f} seconds")

# 4. Generate Association Rules via FP-Growth
rules_fp = association_rules(freq_fpgrowth, metric="confidence", min_threshold=0.50)
print("\nTop 5 Association Rules via FP-Growth:")
print(rules_fp[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())