# Databricks notebook source
# MAGIC %md
# MAGIC # Milk intake forecast and allocation (Databricks version)
# MAGIC
# MAGIC Same pipeline as `python -m milkplan run`, written for Databricks Free Edition:
# MAGIC Eurostat snapshot -> validation -> rolling-origin backtest -> forecast with P10/P50/P90 -> allocation per scenario.
# MAGIC Results are written as Delta tables so a Power BI report can read them.
# MAGIC
# MAGIC **Status:** written for Databricks, not yet run there. The local and CI runs use the same functions.
# MAGIC Import this repo as a Git folder in the workspace, then run all cells.

# COMMAND ----------

# MAGIC %pip install statsmodels scikit-learn scipy

# COMMAND ----------

import os
import sys

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(repo_root, "src"))

from milkplan import allocate, data, forecast  # noqa: E402

# COMMAND ----------

GEO = "DK"
START = "2005-01-01"
ORIGINS = 36
CATALOG_SCHEMA = "workspace.milk_planning"  # change to your catalog.schema

df = data.load(GEO)
y = df.milk_kt.dropna()
y = y[y.index >= START]
checks = data.validate(df.milk_kt, START)
assert not checks["missing_months"] and not checks["negative_values"], checks
checks

# COMMAND ----------

bt = forecast.backtest(y, origins=ORIGINS)
scores = forecast.score(bt)
best = scores.index[0]
display(scores.reset_index())

# COMMAND ----------

fc = forecast.forecast_with_intervals(y, best, bt)
fat_fc = forecast.seasonal_naive(df.fat_pct.dropna(), len(fc))
fc["fat_pct"] = fat_fc.values
plan = allocate.allocate_scenarios(fc, fat_fc)
summary = allocate.summarise(plan)
display(fc)
display(summary)

# COMMAND ----------

# Write Delta tables for Power BI (Databricks connector) when Spark is available.
if "spark" in globals():
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_SCHEMA}")
    tables = {
        "history": df.reset_index(names="month"),
        "backtest": bt,
        "scores": scores.reset_index().melt(id_vars="model", var_name="horizon", value_name="mape_pct"),
        "forecast": fc.assign(model=best),
        "allocation": plan,
        "allocation_summary": summary,
    }
    for name, pdf in tables.items():
        pdf = pdf.copy()
        pdf.columns = [str(c) for c in pdf.columns]
        spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true") \
            .saveAsTable(f"{CATALOG_SCHEMA}.{name}")
    print("Delta tables written to", CATALOG_SCHEMA)
