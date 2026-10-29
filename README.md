# Turnover risk model on Databricks (POC)

A monthly Databricks job that retrains a simple employee-turnover model and scores current employees.
It was built as a proof of concept on Databricks Free Edition with **mock HR data**, and is set up to
deploy unchanged to another workspace.

## What it does

One job, `turnover_monthly`, with two tasks that run on serverless compute:

1. **train** (`src/adm_toy/train.py`)
   - Reads `<catalog>.<schema>.employee_snapshots`.
   - Rebuilds `turnover_periods`: one row per employee per 6-month period over the last 24 months. The outcome
     is whether the employee left during the period; reductions in force are excluded.
   - Fits a logistic regression on binned age and tenure, using the most recent 12 months.
   - Logs an MLflow run, registers a new version of `<catalog>.<schema>.turnover_model`, and points the
     alias `champion` at it.
2. **score** (`src/adm_toy/score.py`), after train succeeds
   - Loads `turnover_model@champion`.
   - Scores everyone employed in the latest snapshot and writes 6- and 12-month risk to
     `turnover_predictions`, replacing any earlier rows for that month.

Schedule: 06:00 America/Chicago on the 2nd of each month. The deploying user gets an email if a run fails.

## Input

Table `<catalog>.<schema>.employee_snapshots`, one row per employee per month employed, with at least
25 month-end snapshots (24 months of history plus the current month):

| Column | Type | Meaning |
|---|---|---|
| `employee_id` | string | Employee identifier |
| `snapshot_date` | date | Month-end date of the snapshot |
| `age` | double | Age in years at `snapshot_date` |
| `tenure` | double | Years since hire at `snapshot_date` |
| `turnover` | int | 1 in the month the employee separated |
| `term_reason` | string | NULL while employed; `Reduction in Force` is excluded from the target |
| `segment`, `region` | string | Carried into the predictions for reporting |

## Outputs

| Object | Kind | Contents |
|---|---|---|
| `<catalog>.<schema>.turnover_periods` | table | Model input, rebuilt on every run |
| `<catalog>.<schema>.turnover_model` | registered model | One version per retrain; alias `champion` |
| `<catalog>.<schema>.turnover_predictions` | table | `employee_id`, `snapshot_date`, `segment`, `region`, `age`, `tenure`, `risk_6m`, `risk_12m`, `model_version`, `scored_at` |
| MLflow experiment (`var.experiment`) | experiment | One run per retrain, with backtest AUC and training turnover rate |

## Deploy

Requirements: the Databricks CLI, and a workspace with Unity Catalog and serverless jobs, a catalog and
schema you can write to, and the input table.

```bash
databricks auth login --host <workspace-url> --profile <name>
databricks bundle validate -t <target> -p <name>
databricks bundle deploy -t <target> -p <name>
databricks bundle run turnover_monthly -t <target> -p <name>   # optional first run
```

`databricks.yml` defines the job. To deploy to another workspace, add a target (a commented
template for ADM is in the file) rather than editing the job itself.

## Operating it

- **Pause or resume:** use the job page's schedule controls, or set `pause_status` in `databricks.yml` and deploy.
- **Re-run a month:** click Run now. Scores for that month are replaced, not duplicated.
- **Roll back the model:** move the `champion` alias to an earlier version, either in Catalog Explorer or with
  `databricks registered-models set-alias <catalog>.<schema>.turnover_model champion <version>`.
  The next scoring run uses it.
- **Change the code:** edit, commit, then `databricks bundle deploy`. The job always runs the deployed copy.

## Open decisions before production

- Which catalog and schema to use, and where `employee_snapshots` comes from.
- Running the job as a service principal instead of a personal user.
- Whether each retrain should become `champion` automatically (current behaviour) or only after review.
- Whether age may be used as a model input.
- When the month-end snapshot lands, so the schedule runs after it.

## Repository layout

```
databricks.yml                  job definition (Databricks Asset Bundle)
src/adm_toy/train.py            task 1: retrain and register
src/adm_toy/score.py            task 2: score and write predictions

POC only: mock data setup
src/adm_toy/datagen.py          seeded mock employee snapshots (runs locally)
sql/00_setup.sql                catalog, schema and landing volume
sql/01_create_tables.sql        loads the CSV into employee_snapshots
scripts/run_sql.py              runs a .sql file on a SQL warehouse via the CLI
notebooks/01_explore_data.py    optional exploration notebook
requirements.txt                local dependencies for datagen.py
```

### Rebuilding the POC data

```bash
python src/adm_toy/datagen.py --out data
python scripts/run_sql.py sql/00_setup.sql
databricks fs cp data/employee_snapshots.csv dbfs:/Volumes/adm_toy/hr/landing/employee_snapshots.csv -p adm-toy
python scripts/run_sql.py sql/01_create_tables.sql
```

`scripts/run_sql.py` uses the CLI profile `adm-toy` by default; pass `-p <name>` for another.
