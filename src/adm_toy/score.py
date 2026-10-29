"""Scoring step of the turnover_monthly job: score everyone employed in the latest snapshot.

Loads the Unity Catalog model through its "champion" alias, so this script never names a version,
and writes 6- and 12-month turnover risk to {catalog}.{schema}.turnover_predictions. Scores for the
same snapshot month are replaced, so re-running a month is safe.

Runs on Databricks as a Python script task (see databricks.yml); it is not meant to run locally.
"""

from __future__ import annotations

import argparse

import mlflow
import pandas as pd
from mlflow import MlflowClient
from pyspark.sql import SparkSession

FEATURES = ["age", "tenure"]

PREDICTIONS_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
  employee_id STRING COMMENT 'Employee identifier',
  snapshot_date DATE COMMENT 'Snapshot month that was scored',
  segment STRING COMMENT 'Job segment at snapshot_date',
  region STRING COMMENT 'Region at snapshot_date',
  age DOUBLE COMMENT 'Age in years at snapshot_date',
  tenure DOUBLE COMMENT 'Tenure in years at snapshot_date',
  risk_6m DOUBLE COMMENT 'Probability of leaving within 6 months',
  risk_12m DOUBLE COMMENT 'Probability of leaving within 12 months: 1 - (1 - risk_6m)^2',
  model_version BIGINT COMMENT 'Version of the registered model that produced the score',
  scored_at TIMESTAMP COMMENT 'When the score was written'
)
COMMENT 'Monthly turnover risk scores for current employees, one row per employee per snapshot month'
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", default="adm_toy")
    parser.add_argument("--schema", default="hr")
    args = parser.parse_args()

    schema = f"{args.catalog}.{args.schema}"
    model_name = f"{schema}.turnover_model"
    table = f"{schema}.turnover_predictions"
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")

    version = MlflowClient().get_model_version_by_alias(model_name, "champion").version
    model = mlflow.sklearn.load_model(f"models:/{model_name}@champion")

    current = spark.sql(f"""
        SELECT employee_id, snapshot_date, segment, region, age, tenure
        FROM {schema}.employee_snapshots
        WHERE snapshot_date = (SELECT max(snapshot_date) FROM {schema}.employee_snapshots)
          AND term_reason IS NULL
    """).toPandas()
    as_of = current["snapshot_date"].iloc[0]

    risk_6m = model.predict_proba(current[FEATURES])[:, 1]
    scores = current.assign(
        risk_6m=risk_6m,
        risk_12m=1 - (1 - risk_6m) ** 2,
        model_version=int(version),
        scored_at=pd.Timestamp.now(tz="UTC"),
    )

    spark.sql(PREDICTIONS_DDL.format(table=table))
    (
        spark.createDataFrame(scores)
        .write.mode("overwrite")
        .option("replaceWhere", f"snapshot_date = DATE'{as_of}'")
        .saveAsTable(table)
    )
    print(
        f"scored {len(scores):,} employees as of {as_of} with {model_name} v{version}: "
        f"expected leavers in 12 months {scores['risk_12m'].sum():,.0f} ({scores['risk_12m'].mean():.1%})"
    )


if __name__ == "__main__":
    main()
