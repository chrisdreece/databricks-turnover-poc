# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Explore the employee snapshots
# MAGIC
# MAGIC Reads `adm_toy.hr.employee_snapshots` (created by `sql/01_create_tables.sql`) from both SQL and Python.
# MAGIC
# MAGIC Every table is addressed as `catalog.schema.table`. The same name works in a `%sql` cell, in
# MAGIC `spark.table(...)`, from a SQL warehouse, or from a BI tool, and Unity Catalog applies the same permissions everywhere.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Headcount and separations by month. Note the reduction in force in January 2026.
# MAGIC SELECT snapshot_date,
# MAGIC        count(*) AS headcount,
# MAGIC        sum(turnover) AS turnover,
# MAGIC        sum(CASE WHEN term_reason = 'Reduction in Force' THEN 1 ELSE 0 END) AS rif,
# MAGIC        sum(retirement) AS retirements
# MAGIC FROM adm_toy.hr.employee_snapshots
# MAGIC GROUP BY snapshot_date
# MAGIC ORDER BY snapshot_date

# COMMAND ----------

# MAGIC %md
# MAGIC Every Delta write is a new table version. Version 0 is the empty `CREATE`, version 1 the load, and older versions stay queryable ("time travel").

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'version 0' AS version, count(*) AS row_count FROM adm_toy.hr.employee_snapshots VERSION AS OF 0
# MAGIC UNION ALL
# MAGIC SELECT 'current', count(*) FROM adm_toy.hr.employee_snapshots

# COMMAND ----------

import json

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

snapshots = spark.table("adm_toy.hr.employee_snapshots")

# Spark DataFrames are lazy and run on Databricks compute; toPandas() pulls the result into the notebook's memory.
df = snapshots.toPandas()
df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monthly turnover rate by tenure and age (RIF excluded)

# COMMAND ----------

df["voluntary_or_involuntary"] = np.where(df["term_reason"] == "Reduction in Force", 0, df["turnover"])
df["tenure_bin"] = pd.cut(df["tenure"], [-np.inf, 0.5, 1, 2, 3, 5, 9, 15, 20, np.inf]).astype(str)
df["age_bin"] = pd.cut(df["age"], [-np.inf, 28, 33, 37, 47, 52, 60, 65, np.inf]).astype(str)

by_tenure = df.groupby("tenure_bin", observed=True).agg(employee_months=("employee_id", "size"), monthly_rate=("voluntary_or_involuntary", "mean"))
display(by_tenure.reset_index())

# COMMAND ----------

by_age = df.groupby("age_bin", observed=True).agg(employee_months=("employee_id", "size"), monthly_rate=("voluntary_or_involuntary", "mean"))
display(by_age.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data checks

# COMMAND ----------

last_month = snapshots.agg(F.max("snapshot_date")).first()[0]
leavers = snapshots.filter((F.col("turnover") == 1) | (F.col("retirement") == 1)).select("employee_id", F.col("snapshot_date").alias("left_on"))
checks = {
    "rows": snapshots.count(),
    "employees": snapshots.select("employee_id").distinct().count(),
    "months": snapshots.select("snapshot_date").distinct().count(),
    "current_month": str(last_month),
    "current_headcount_excl_leavers": snapshots.filter((F.col("snapshot_date") == last_month) & (F.col("turnover") == 0) & (F.col("retirement") == 0)).count(),
    "duplicate_employee_months": snapshots.count() - snapshots.select("employee_id", "snapshot_date").distinct().count(),
    "rows_after_separation": snapshots.join(leavers, "employee_id").filter(F.col("snapshot_date") > F.col("left_on")).count(),
    "separations_without_reason": snapshots.filter(((F.col("turnover") == 1) | (F.col("retirement") == 1)) & F.col("term_reason").isNull()).count(),
    "rif_rows": snapshots.filter(F.col("term_reason") == "Reduction in Force").count(),
}
checks

# COMMAND ----------

dbutils.notebook.exit(json.dumps(checks))
