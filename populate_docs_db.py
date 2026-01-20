import sys
import os
from database import session_manager

# Essential Documentation to Seed
docs = [
    {
        "library": "Pandas",
        "topic": "Handling Missing Data",
        "content": """
To handle missing data in Pandas:
1. Check for missing values: `df.isnull().sum()`
2. Drop missing values: `df.dropna(subset=['col_name'])` or `df.dropna()` (use carefully)
3. Fill missing values: `df['col'].fillna(value)` or `df['col'].fillna(df['col'].mean())`
4. Interpolate: `df.interpolate(method='linear')`

Best Practice: Always report the number of missing values before and after cleaning.
"""
    },
    {
        "library": "Matplotlib",
        "topic": "Agg Backend & Saving",
        "content": """
In this environment, Matplotlib must use the 'Agg' backend to prevent GUI errors.

Standard Pattern:
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

plt.figure(figsize=(10, 6))
# ... plotting code ...

# Save to Base64
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight')
buf.seek(0)
print('IMAGE_BASE64:' + base64.b64encode(buf.read()).decode())
plt.clf()
```
"""
    },
    {
        "library": "Seaborn",
        "topic": "Styling",
        "content": "Use `sns.set_theme(style='whitegrid')` for clean plots. Common palettes: 'viridis' (sequential), 'Set2' (categorical)."
    }
]

def seed_docs():
    print("📘 Seeding Documentation Knowledge Base...")
    for d in docs:
        session_manager.save_documentation(d["library"], d["topic"], d["content"])
    print(f"✅ Added {len(docs)} documentation entries.")

if __name__ == "__main__":
    seed_docs()