"""
Launch Datasette visualization for truck sales data.
"""
import sqlite3
import subprocess
import pandas as pd

# Create SQLite database from CSV
df = pd.read_csv("truck_sales_targets.csv")

with sqlite3.connect("truck_sales.db") as conn:
    df.to_sql("truck_sales", conn, if_exists="replace", index=False)

print(f"Created database with {len(df):,} records")

# Launch Datasette
subprocess.run(["datasette", "truck_sales.db"])
