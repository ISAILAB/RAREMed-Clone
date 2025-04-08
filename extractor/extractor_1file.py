import re
import pandas as pd
import sys
import os

input_arg = sys.argv[1]

# If input is a folder, append .log automatically
if not input_arg.endswith(".log"):
    input_arg = input_arg + "/" + input_arg + ".log"

log_path = os.path.join("../src/log/mimic-iii/RAREMed", input_arg)

# Check existence
if not os.path.exists(log_path):
    print(f"❌ File not found: {log_path}")
    sys.exit(1)

column_order = [
    "weighted_Jaccard",
    "corr",
    "slope_corr",
    "Jaccard",
    "DDI Rate",
    "PRAUC",
    "AVG_F1",
    "AVG_PRC",
    "AVG_RECALL",
    "AVG_MED",
    "loss_train_all",
    "loss_train_bce",
    "loss_train_multi",
    "loss_train_ddi",
    "loss_val_all",
    "loss_val_bce",
    "loss_val_multi",
    "loss_val_ddi",
    "best_epoch",
    "best_ja",
]

results = []
current = {}

with open(log_path, "r") as file:
    lines = file.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()

    if "weighted_Jaccard" in line:
        match = re.search(
            r"weighted_Jaccard:\s([\d.]+), corr:\s([\d.]+), slope_corr:\s([\d.]+)", line
        )
        if match:
            current["weighted_Jaccard"] = float(match.group(1))
            current["corr"] = float(match.group(2))
            current["slope_corr"] = float(match.group(3))

    elif "Epoch" in line and "Jaccard:" in line:
        match = re.search(
            r"Jaccard:\s([\d.]+), DDI Rate:\s([\d.]+), PRAUC:\s([\d.]+), AVG_F1:\s([\d.]+), "
            r"AVG_PRC:\s([\d.]+), AVG_RECALL:\s([\d.]+), AVG_MED:\s([\d.]+)",
            line,
        )
        if match:
            current["Jaccard"] = float(match.group(1))
            current["DDI Rate"] = float(match.group(2))
            current["PRAUC"] = float(match.group(3))
            current["AVG_F1"] = float(match.group(4))
            current["AVG_PRC"] = float(match.group(5))
            current["AVG_RECALL"] = float(match.group(6))
            current["AVG_MED"] = float(match.group(7))

    elif "loss_train_all" in line:
        # Ghép dòng tiếp theo nếu cần
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        full_line = line + " " + next_line
        match = re.search(
            r"loss_train_all:([\d.]+), loss_train_bce:\s*([\d.]+), loss_train_multi:([\d.]+), loss_train_ddi:([\d.]+)\s*"
            r"loss_val_all:\s*([\d.]+), loss_val_bce:\s*([\d.]+), loss_val_multi:\s*([\d.]+), loss_val_ddi:\s*([\d.]+)",
            full_line,
        )
        if match:
            current["loss_train_all"] = float(match.group(1))
            current["loss_train_bce"] = float(match.group(2))
            current["loss_train_multi"] = float(match.group(3))
            current["loss_train_ddi"] = float(match.group(4))
            current["loss_val_all"] = float(match.group(5))
            current["loss_val_bce"] = float(match.group(6))
            current["loss_val_multi"] = float(match.group(7))
            current["loss_val_ddi"] = float(match.group(8))
        i += 1  # bỏ qua dòng tiếp theo vì đã dùng

    elif "best_epoch" in line:
        match = re.search(r"best_epoch:\s(\d+), best_ja:\s([\d.]+)", line)
        if match:
            current["best_epoch"] = int(match.group(1))
            current["best_ja"] = float(match.group(2))
            results.append(current)
            current = {}

    i += 1  # tiến đến dòng tiếp theo

# Save results
os.makedirs("outputs", exist_ok=True)
output_path = os.path.join("outputs", os.path.basename(input_arg) + ".csv")

df = pd.DataFrame(results)

# Add missing columns with None values
for col in column_order:
    if col not in df.columns:
        df[col] = None

# Reorder columns
df = df[column_order]

# Save with tab-separated values
df.to_csv(output_path, index=False, sep="\t")

print(f"✅ CSV saved to {output_path}")
print(df)
