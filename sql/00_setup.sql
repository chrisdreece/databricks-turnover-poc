-- Phase 1, step 1: namespace for the project.
-- Unity Catalog is a three-level hierarchy: catalog.schema.object, where objects are tables, views, volumes, models and functions.
-- Run with: python scripts/run_sql.py sql/00_setup.sql
-- Note: on Free Edition, CREATE CATALOG works in SQL (it uses Default Storage) but the REST/CLI `catalogs create` call does not.

CREATE CATALOG IF NOT EXISTS adm_toy
  COMMENT 'Toy turnover-prediction project (mock data only)';

CREATE SCHEMA IF NOT EXISTS adm_toy.hr
  COMMENT 'Employee snapshots, turnover model inputs and predictions';

-- A volume is governed storage for files (CSV, images, etc.) that are not tables. Raw uploads land here.
CREATE VOLUME IF NOT EXISTS adm_toy.hr.landing
  COMMENT 'Raw CSV files from src/adm_toy/datagen.py';
