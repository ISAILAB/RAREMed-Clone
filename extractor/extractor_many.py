import re
import pandas as pd
import sys
import os

input_files = sys.argv[1:]

if not input_files:
    print("❌ Please provide at least one log file name.")
    sys.exit(1)

# Define desired column order
column_order = [
    "weighted_Jaccard", "corr", "slope_corr",
    "Jaccard", "DDI Rate", "PRAUC", "AVG_F1",
    "AVG_PRC", "AVG_RECALL", "AVG_MED",
    "best_epoch", "best_ja"
]

for input_name in input_files:
    log_path = os.path.join('../src/log/mimic-iii/RAREMed', input_name, input_name + '.log')

    if not os.path.exists(log_path):
        print(f"⚠️  File not found: {log_path}")
        continue

    results = []
    current = {}

    with open(log_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if 'weighted_Jaccard' in line:
            match = re.search(r'weighted_Jaccard:\s([\d.]+), corr:\s([\d.]+), slope_corr:\s([\d.]+)', line)
            if match:
                current['weighted_Jaccard'] = float(match.group(1))
                current['corr'] = float(match.group(2))
                current['slope_corr'] = float(match.group(3))

        elif 'Epoch' in line and 'Jaccard:' in line:
            match = re.search(
                r'Jaccard:\s([\d.]+), DDI Rate:\s([\d.]+), PRAUC:\s([\d.]+), AVG_F1:\s([\d.]+), '
                r'AVG_PRC:\s([\d.]+), AVG_RECALL:\s([\d.]+), AVG_MED:\s([\d.]+)', line)
            if match:
                current['Jaccard'] = float(match.group(1))
                current['DDI Rate'] = float(match.group(2))
                current['PRAUC'] = float(match.group(3))
                current['AVG_F1'] = float(match.group(4))
                current['AVG_PRC'] = float(match.group(5))
                current['AVG_RECALL'] = float(match.group(6))
                current['AVG_MED'] = float(match.group(7))

        elif 'best_epoch' in line:
            match = re.search(r'best_epoch:\s(\d+), best_ja:\s([\d.]+)', line)
            if match:
                current['best_epoch'] = int(match.group(1))
                current['best_ja'] = float(match.group(2))
                results.append(current)
                current = {}

    if results:
        os.makedirs('many_output', exist_ok=True)
        df = pd.DataFrame(results)

        # Ensure all columns are present, even if empty
        for col in column_order:
            if col not in df.columns:
                df[col] = None

        # Reorder columns
        df = df[column_order]

        output_csv = os.path.join('many_output', input_name + '.csv')
        df.to_csv(output_csv, index=False, sep='\t')
        print(f"✅ Extracted to: {output_csv}")
    else:
        print(f"⚠️  No data extracted from {input_name}")
