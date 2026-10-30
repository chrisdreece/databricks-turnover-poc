-- Phase 1, step 1: namespace for the project.
-- Unity Catalog is a three-level hierarchy: catalog.schema.object, where objects are tables, views, volumes, models and functions.
-- Run with: python scripts/run_sql.py sql/00_setup.sql
-- The catalog is created once, outside this script, with its files in your own storage account:
--   CREATE CATALOG adm_toy MANAGED LOCATION 'abfss://<container>@<storage-account>.dfs.core.windows.net/adm_toy';
-- (On Free Edition it can instead be created without a location, using Databricks' default storage.)

CREATE SCHEMA IF NOT EXISTS adm_toy.hr
  COMMENT 'Employee snapshots, turnover model inputs and predictions';

-- A volume is governed storage for files (CSV, images, etc.) that are not tables. Raw uploads land here.
CREATE VOLUME IF NOT EXISTS adm_toy.hr.landing
  COMMENT 'Raw CSV files from src/adm_toy/datagen.py';
