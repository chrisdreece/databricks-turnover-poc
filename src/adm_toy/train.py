"""Training step of the turnover_monthly job: refit the age + tenure model and make it the champion.

1. Rebuild the model table (one row per employee per 6-month period) from the monthly snapshots.
2. Fit a logistic regression on binned age and tenure using the two most recent periods.
3. Log the run to MLflow, register it as a new version of the Unity Catalog model, and point
   the "champion" alias at that version so the scoring step picks it up.

Runs on Databricks as a Python script task (see databricks.yml); it is not meant to run locally.
"""

from __future__ import annotations

import argparse

import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

FEATURES = ["age", "tenure"]
TARGET = "left_in_period"
TENURE_EDGES = [0.5, 1, 2, 3, 4, 5, 9, 15, 20]
AGE_EDGES = [28, 30, 33, 37, 47, 52, 60, 65]

# One row per employee present in the first month of each 6-month period; left_in_period = 1 if they
# turned over during it. Reductions in force don't count, and the latest (current) month is held out.
PERIODS_SQL = """
CREATE OR REPLACE TABLE {schema}.turnover_periods
COMMENT 'Model table: employees at the start of each 6-month period and whether they left during it'
AS
WITH history AS (
  SELECT *,
         CASE WHEN term_reason = 'Reduction in Force' THEN 0 ELSE turnover END AS turnover_excl_rif,
         dense_rank() OVER (ORDER BY snapshot_date) AS month_rank
  FROM {schema}.employee_snapshots
  WHERE snapshot_date < (SELECT max(snapshot_date) FROM {schema}.employee_snapshots)
    AND snapshot_date >= add_months((SELECT max(snapshot_date) FROM {schema}.employee_snapshots), -24)
),
with_period AS (
  SELECT *,
         CAST(ceil(month_rank / 6.0) AS INT) AS period,
         max(turnover_excl_rif) OVER (PARTITION BY employee_id, CAST(ceil(month_rank / 6.0) AS INT)) AS left_in_period
  FROM history
)
SELECT employee_id, period, snapshot_date AS period_start, age, tenure, left_in_period
FROM with_period
WHERE month_rank IN (1, 7, 13, 19)
"""


def binned(column, edges):
    return (
        f"{column}_bin",
        Pipeline([
            ("bin", FunctionTransformer(np.digitize, kw_args={"bins": edges, "right": True}, feature_names_out="one-to-one")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]),
        [column],
    )


def make_model():
    """Bins live inside the pipeline, so scoring passes raw age and tenure. C=1e6 makes it a plain (unpenalized) GLM."""
    return Pipeline([
        ("bins", ColumnTransformer([binned("tenure", TENURE_EDGES), binned("age", AGE_EDGES)])),
        ("glm", LogisticRegression(C=1e6, max_iter=1000)),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", default="adm_toy")
    parser.add_argument("--schema", default="hr")
    parser.add_argument("--experiment", required=True, help="MLflow experiment path")
    args = parser.parse_args()

    schema = f"{args.catalog}.{args.schema}"
    model_name = f"{schema}.turnover_model"
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(args.experiment)

    spark.sql(PERIODS_SQL.format(schema=schema))
    periods = spark.table(f"{schema}.turnover_periods").toPandas()
    as_of = str(spark.sql(f"SELECT max(snapshot_date) FROM {schema}.employee_snapshots").first()[0])
    train = periods[periods["period"].isin([3, 4])]

    with mlflow.start_run(run_name=f"train as of {as_of}") as run:
        # Quality check before refitting: fit on the older year, score the newer one.
        check = make_model().fit(periods.loc[periods["period"] <= 2, FEATURES], periods.loc[periods["period"] <= 2, TARGET])
        newer = periods[periods["period"] >= 3]
        backtest_auc = roc_auc_score(newer[TARGET], check.predict_proba(newer[FEATURES])[:, 1])

        model = make_model().fit(train[FEATURES], train[TARGET])
        mlflow.log_params({
            "snapshot_as_of": as_of,
            "inputs": ",".join(FEATURES),
            "train_periods": "3,4",
            "tenure_edges": TENURE_EDGES,
            "age_edges": AGE_EDGES,
        })
        mlflow.log_metrics({
            "train_rows": len(train),
            "train_6m_turnover_rate": float(train[TARGET].mean()),
            "backtest_auc": float(backtest_auc),
        })
        info = mlflow.sklearn.log_model(
            model,
            "model",
            signature=infer_signature(train[FEATURES], model.predict_proba(train[FEATURES])),
            input_example=train[FEATURES].head(5),
            registered_model_name=model_name,
        )

    client = MlflowClient()
    version = getattr(info, "registered_model_version", None) or max(
        int(v.version) for v in client.search_model_versions(f"name = '{model_name}'")
    )
    client.set_registered_model_alias(model_name, "champion", version)
    print(f"run {run.info.run_id}: backtest AUC {backtest_auc:.3f}; registered {model_name} version {version} as champion")


if __name__ == "__main__":
    main()
