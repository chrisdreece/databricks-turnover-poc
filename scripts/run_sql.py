"""Run SQL statements against a Databricks SQL warehouse through the Databricks CLI.

Statements are separated by semicolons at line ends; `--` comment lines are skipped.

Usage:
    python scripts/run_sql.py sql/01_create_tables.sql
    python scripts/run_sql.py -e "SELECT count(*) FROM adm_toy.logistics.shipments"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_CLI = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "databricks-cli" / "databricks.exe"


def find_cli():
    return shutil.which("databricks") or str(DEFAULT_CLI)


def api(cli, profile, method, path, body=None):
    cmd = [cli, "api", method, path, "-p", profile]
    if body is not None:
        cmd += ["--json", json.dumps(body)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout or "{}")


def default_warehouse(cli, profile):
    warehouses = api(cli, profile, "get", "/api/2.0/sql/warehouses").get("warehouses", [])
    if not warehouses:
        raise RuntimeError("no SQL warehouse found in this workspace")
    return warehouses[0]["id"]


def run(cli, profile, warehouse_id, statement):
    resp = api(cli, profile, "post", "/api/2.0/sql/statements", {
        "warehouse_id": warehouse_id,
        "statement": statement,
        "wait_timeout": "50s",
        "on_wait_timeout": "CONTINUE",
    })
    while resp["status"]["state"] in ("PENDING", "RUNNING"):
        time.sleep(3)
        resp = api(cli, profile, "get", f"/api/2.0/sql/statements/{resp['statement_id']}")
    if resp["status"]["state"] != "SUCCEEDED":
        raise RuntimeError(resp["status"].get("error", {}).get("message", resp["status"]["state"]))
    return resp


def print_result(resp, max_rows=50):
    columns = [c["name"] for c in resp.get("manifest", {}).get("schema", {}).get("columns", [])]
    rows = resp.get("result", {}).get("data_array", []) or []
    if not columns:
        return
    print(" | ".join(columns))
    for row in rows[:max_rows]:
        print(" | ".join("NULL" if v is None else str(v) for v in row))
    if len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} more rows")


def split_statements(text):
    lines = [line for line in text.splitlines() if not line.strip().startswith("--")]
    statements, current = [], []
    for line in lines:
        current.append(line)
        if line.rstrip().endswith(";"):
            statements.append("\n".join(current).strip().rstrip(";"))
            current = []
    if "".join(current).strip():
        statements.append("\n".join(current).strip())
    return [s for s in statements if s]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", type=Path)
    parser.add_argument("-e", "--execute", help="SQL text to run instead of a file")
    parser.add_argument("-p", "--profile", default="adm-toy")
    parser.add_argument("-w", "--warehouse-id")
    args = parser.parse_args()
    if not (args.file or args.execute):
        parser.error("give a .sql file or -e")

    cli = find_cli()
    warehouse_id = args.warehouse_id or default_warehouse(cli, args.profile)
    text = args.execute or args.file.read_text(encoding="utf-8")
    for statement in split_statements(text):
        print(f"\n>>> {statement.splitlines()[0][:100]}")
        try:
            print_result(run(cli, args.profile, warehouse_id, statement))
        except RuntimeError as err:
            print(f"FAILED: {err}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
