import os
import pandas as pd

# Instead of using settings.BASE_DIR, just look directly at your data directory path
csv_path = "data.csv"
df = pd.read_csv(csv_path)

def is_uncertain(val):
    if pd.isna(val):
        return True
    try:
        v = int(float(val))
        return v == 1
    except (ValueError, TypeError):
        val_str = str(val).strip().lower()
        return 'signal' not in val_str and 'noise' not in val_str

# 3. Create a boolean column tracking whether the item is uncertain
df['is_uncertain'] = df['ai_classification'].apply(is_uncertain)

grouped_df = df.groupby(['dprime_ai', 'dprime_human', 'thresholds_distance']).agg(
    total_items=('item_id', 'count'),
    uncertain_count=('is_uncertain', 'sum')
).reset_index()

grouped_df['uncertain_percentage'] = (grouped_df['uncertain_count'] / grouped_df['total_items']) * 100

print(grouped_df[['dprime_ai', 'dprime_human', 'thresholds_distance', 'uncertain_percentage']])