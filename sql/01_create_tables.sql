-- Phase 1, step 2: load the raw CSV from the landing volume into a Delta table.
-- The schema is declared explicitly rather than inferred, so a bad file fails loudly instead of changing column types.
-- Run with: python scripts/run_sql.py sql/01_create_tables.sql

CREATE OR REPLACE TABLE adm_toy.hr.employee_snapshots (
  employee_id STRING COMMENT 'Employee identifier',
  snapshot_date DATE COMMENT 'Month-end date of the snapshot',
  segment STRING COMMENT 'Job segment',
  region STRING COMMENT 'Central, East, South or West',
  employment_status STRING COMMENT 'Full-Time, Part-Time or Leave',
  hire_date DATE COMMENT 'Most recent hire date',
  age DOUBLE COMMENT 'Age in years at snapshot_date',
  tenure DOUBLE COMMENT 'Years since hire_date at snapshot_date',
  annual_salary DOUBLE COMMENT 'Annualized base pay',
  turnover INT COMMENT '1 if the employee separated this month (voluntary, involuntary or RIF)',
  retirement INT COMMENT '1 if the employee retired this month',
  term_reason STRING COMMENT 'Voluntary, Involuntary, Retirement or Reduction in Force; NULL if still employed'
)
COMMENT 'Monthly employee snapshots: one row per employee per month employed (raw)';

INSERT INTO adm_toy.hr.employee_snapshots
SELECT employee_id, snapshot_date, segment, region, employment_status, hire_date, age, tenure,
       annual_salary, turnover, retirement, term_reason
FROM read_files(
  '/Volumes/adm_toy/hr/landing/employee_snapshots.csv',
  format => 'csv', header => true,
  schema => 'employee_id STRING, snapshot_date DATE, segment STRING, region STRING, employment_status STRING, hire_date DATE, age DOUBLE, tenure DOUBLE, annual_salary DOUBLE, turnover INT, retirement INT, term_reason STRING'
);
