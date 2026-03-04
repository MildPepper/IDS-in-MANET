import pandas as pd
import numpy as np

print(">>> Step 2: Optimized Feature Engineering (v3 with IDs)...")

# 1. Load the cleaned data
try:
    # Keep Time/NodeID/Scenario/Run for later IDS scoring; do NOT use as model features.
    df = pd.read_csv("combined_with_ids.csv")
    print(f"Loaded dataset with {len(df)} rows.")
except FileNotFoundError:
    print("ERROR: 'combined_with_ids.csv' not found. Run merge.py first.")
    exit()

epsilon = 1e-6

# --- Existing Features ---
df['Drop_Ratio'] = df['Pkts_Dropped'] / (df['Pkts_Forwarded'] + df['Pkts_Dropped'] + epsilon)
df['Forward_Ratio'] = df['Pkts_Forwarded'] / (df['Pkts_Forwarded'] + df['Pkts_Dropped'] + epsilon)
df['RREP_per_RREQ'] = df['RREP_Sent'] / (df['RREQ_Recv'] + epsilon)
df['Control_Overhead'] = df['RREQ_Sent'] + df['RREQ_Recv'] + df['RREP_Sent'] + df['RREP_Recv']
df['Attraction_Score'] = df['RREP_Sent'] * df['Drop_Ratio']
df['Net_Flow'] = df['Pkts_Forwarded'] - df['Pkts_Dropped']

# --- NEW: Log Transformation for Model B (Autoencoder) ---
# This keeps the "signal" but makes the numbers manageable (0 to ~19 instead of 0 to 200M)
df['RREP_per_RREQ_log'] = np.log1p(df['RREP_per_RREQ'])
df['Attraction_Score_log'] = np.log1p(df['Attraction_Score'])

# 3. Save
output_filename = "engineered_dataset_v3.csv"
df.to_csv(output_filename, index=False)

print(f"SUCCESS! Saved to: {output_filename}")
