import json
import re
from collections import defaultdict

# 👉 tera notebook path
notebook_path = "notebooks/churn_analysis.ipynb"

# 👉 output file (root folder me banega)
output_file = "column_usage.txt"

# notebook load kar
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# store karne ke liye dict
column_usage = defaultdict(set)   # set use kiya → duplicate avoid

# regex patterns
patterns = [
    r"df\[['\"](.*?)['\"]\]",
    r"X\[['\"](.*?)['\"]\]",
    r"x\s*=\s*['\"](.*?)['\"]",
    r"y\s*=\s*['\"](.*?)['\"]",
    r"hue\s*=\s*['\"](.*?)['\"]"
]

# cells loop
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])

        for pattern in patterns:
            matches = re.findall(pattern, code)
            for col in matches:
                column_usage[col].add(code.strip())

# file me write
with open(output_file, "w", encoding="utf-8") as f:
    for col, usages in column_usage.items():
        f.write(f"\n===== Column: {col} =====\n")
        for i, usage in enumerate(usages, 1):
            f.write(f"\n[{i}] {usage}\n")

print("✅ Done! Output saved in:", output_file)
