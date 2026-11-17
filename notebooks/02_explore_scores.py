# Databricks notebook source
# MAGIC %md
# MAGIC # Explore the data and the scores
# MAGIC
# MAGIC A scratchpad. Run one cell at a time with the play button, or Shift+Enter.
# MAGIC Nothing here is part of the monthly job.

# COMMAND ----------

# Load the snapshot table into pandas, like reading a CSV into a data frame.
df = spark.table("adm_toy.hr.employee_snapshots").toPandas()
df.shape

# COMMAND ----------

# Descriptives for age and tenure.
df[["age", "tenure", "annual_salary"]].describe()

# COMMAND ----------

# Headcount and separations by month.
(
    df.groupby("snapshot_date")
    .agg(headcount=("employee_id", "size"), turnover=("turnover", "sum"), retirement=("retirement", "sum"))
    .reset_index()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The scores the job produced

# COMMAND ----------

scores = spark.table("adm_toy.hr.turnover_predictions").toPandas()
scores.sort_values("risk_12m", ascending=False).head(20)

# COMMAND ----------

# Average 12-month risk by segment.
scores.groupby("segment")["risk_12m"].agg(["count", "mean"]).sort_values("mean", ascending=False)

# COMMAND ----------

# A quick chart: distribution of 12-month risk.
scores["risk_12m"].plot(kind="hist", bins=40, title="12-month turnover risk")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The same thing in SQL
# MAGIC The `%sql` line below turns a cell into SQL.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT segment, region, round(avg(risk_12m), 3) AS avg_risk_12m, count(*) AS employees
# MAGIC FROM adm_toy.hr.turnover_predictions
# MAGIC GROUP BY segment, region
# MAGIC ORDER BY avg_risk_12m DESC
