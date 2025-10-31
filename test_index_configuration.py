#!/usr/bin/env python3
"""
Evaluate a fixed index configuration on a given workload and benchmark database.

Examples
--------
- Using a file with one index per line:
    python scripts/test_index_configuration.py \
        --benchmark tpch \
        --workload-sql query_files/TPCH/my_workload.sql \
        --index-config-file indexes.txt

- Inline index spec with per-query breakdown:
    python scripts/test_index_configuration.py \
        --benchmark tpcds \
        --workload-sql workloads/tpcds_eval.sql \
        --indexes "{I(C store_sales.ss_item_sk), I(C date_dim.d_date_sk,C date_dim.d_month_seq)}" \
        --per-query --analyze
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Sequence

import sqlparse

from balance.schema import Schema
from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.index import Index
from index_selection_evaluation.selection.workload import Column, Query, Workload

DEFAULT_DATABASES = {
    "tpch": "indexselection_tpch___1",
    "tpchc": "indexselection_tpch___1",
    "tpcds": "indexselection_tpcds___10",
    "tpcdsc": "indexselection_tpcds___10",
    "ceb": "indexselection_job___1",
    "job": "indexselection_job___1",
}

DEFAULT_SCALE_FACTORS = {
    "tpch": 1,
    "tpchc": 1,
    "tpcds": 10,
    "tpcdsc": 10,
    "ceb": 1,
    "job": 1,
}

INDEX_PATTERN = re.compile(r"I\s*\([^()]*\)", re.IGNORECASE)
COLUMN_PATTERN = re.compile(r"C\s+([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an index configuration on a workload.")
    parser.add_argument("--benchmark", required=True, help="Benchmark name (e.g., tpch, tpcds, job).")
    parser.add_argument("--workload-sql", required=True, type=Path, help="Path to SQL file defining the workload.")
    parser.add_argument(
        "--index-config-file",
        type=Path,
        help="Optional text file with one index per line (e.g., `I(C table.col)` format).",
    )
    parser.add_argument(
        "--indexes",
        help="Inline index specification (e.g., `{I(C a.b), I(C c.d,C c.e)}`); indexes within braces are detected automatically.",
    )
    parser.add_argument(
        "--database",
        help="Override database name. Defaults are derived from the benchmark.",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        help="Override scale factor; defaults follow benchmark conventions.",
    )
    parser.add_argument(
        "--cost-mode",
        choices=("whatif", "actual_runtimes"),
        default="whatif",
        help="Use PostgreSQL HypoPG estimation (`whatif`, default) or execute queries (`actual_runtimes`).",
    )
    parser.add_argument(
        "--per-query",
        action="store_true",
        help="Print per-query cost breakdown for baseline vs. indexed configurations.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run ANALYZE on the target database before evaluation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def normalize_benchmark(name: str) -> str:
    return name.strip().lower()


def resolve_database(benchmark: str, explicit: str = None) -> str:
    if explicit:
        return explicit
    if benchmark not in DEFAULT_DATABASES:
        raise ValueError(f"No default database mapping for benchmark '{benchmark}'. Provide --database explicitly.")
    return DEFAULT_DATABASES[benchmark]


def resolve_scale_factor(benchmark: str, explicit: int = None) -> int:
    if explicit is not None:
        return explicit
    if benchmark not in DEFAULT_SCALE_FACTORS:
        raise ValueError(f"No default scale factor for benchmark '{benchmark}'. Provide --scale-factor explicitly.")
    return DEFAULT_SCALE_FACTORS[benchmark]


def load_index_specs(index_file: Path = None, inline_spec: str = None) -> List[str]:
    specs: List[str] = []

    def _extract(raw: str) -> None:
        cleaned = raw.replace("{", " ").replace("}", " ")
        for match in INDEX_PATTERN.finditer(cleaned):
            token = match.group()
            token = re.sub(r"\s+", " ", token).strip()
            token = token.replace("I (", "I(")
            specs.append(token)

    if index_file:
        for line in index_file.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            _extract(stripped)

    if inline_spec:
        _extract(inline_spec)

    return specs


def build_column_lookup(columns: Sequence[Column]) -> Dict[str, Column]:
    lookup: Dict[str, Column] = {}
    for column in columns:
        key = f"{column.table.name}.{column.name}"
        lookup[key] = column
    return lookup


def parse_index(spec: str, column_lookup: Dict[str, Column]) -> Index:
    inner = spec[spec.index("(") + 1 : spec.rindex(")")]
    matches = COLUMN_PATTERN.findall(inner)
    if not matches:
        raise ValueError(f"Could not parse columns from index specification '{spec}'.")
    columns: List[Column] = []
    seen = set()
    for table_name, column_name in matches:
        key = f"{table_name.lower()}.{column_name.lower()}"
        if key not in column_lookup:
            raise KeyError(f"Unknown column '{key}' in index specification '{spec}'.")
        if key in seen:
            continue
        seen.add(key)
        columns.append(column_lookup[key])
    return Index(columns)


def infer_query_columns(query: Query, columns: Sequence[Column], benchmark: str) -> None:
    query_lower = query.text.lower()
    if benchmark != "job":
        for column in columns:
            if column.name in query_lower:
                query.columns.append(column)
    else:
        if "where" not in query.text.lower():
            return
        before_where, _, after_where = query.text.lower().partition("where")
        for column in columns:
            if column.name in after_where and f"{column.table.name} " in before_where:
                query.columns.append(column)


def load_workload(sql_path: Path, schema_columns: Sequence[Column], benchmark: str) -> Workload:
    raw_sql = sql_path.read_text()
    statements = [stmt.strip() for stmt in sqlparse.split(raw_sql) if stmt.strip()]
    queries: List[Query] = []
    for idx, statement in enumerate(statements, start=1):
        query = Query(idx, statement, frequency=1)
        infer_query_columns(query, schema_columns, benchmark)
        queries.append(query)
    description = f"{sql_path.name} ({len(queries)} queries)"
    return Workload(queries, description=description)


def evaluate_cost(
    workload: Workload,
    indexes: Sequence[Index],
    connector: PostgresDatabaseConnector,
    cost_mode: str,
    per_query: bool = False,
):
    baseline_eval = CostEvaluation(connector, cost_estimation=cost_mode)
    if per_query:
        baseline_cost, _, baseline_costs = baseline_eval.calculate_cost_and_plans(workload, [], store_size=False)
    else:
        baseline_cost = baseline_eval.calculate_cost(workload, [], store_size=False)
        baseline_costs = [baseline_cost] * len(workload.queries)
    baseline_eval.complete_cost_estimation()

    indexed_eval = CostEvaluation(connector, cost_estimation=cost_mode)
    if per_query:
        indexed_cost, _, indexed_costs = indexed_eval.calculate_cost_and_plans(workload, indexes, store_size=True)
    else:
        indexed_cost = indexed_eval.calculate_cost(workload, indexes, store_size=True)
        indexed_costs = [indexed_cost] * len(workload.queries)
    indexed_eval.complete_cost_estimation()

    return baseline_cost, indexed_cost, baseline_costs, indexed_costs


def main() -> None:
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")

    benchmark = normalize_benchmark(args.benchmark)
    database_name = resolve_database(benchmark, args.database)
    scale_factor = resolve_scale_factor(benchmark, args.scale_factor)

    index_specs = load_index_specs(args.index_config_file, args.indexes)
    if not index_specs:
        logging.warning("No indexes specified. Evaluation will compare baseline workload only.")
    logging.info("Loaded %d index specification(s).", len(index_specs))

    schema = Schema(args.benchmark, scale_factor, database_name)
    column_lookup = build_column_lookup(schema.columns)
    indexes = [parse_index(spec, column_lookup) for spec in index_specs]

    workload = load_workload(args.workload_sql, schema.columns, benchmark)
    logging.info("Workload '%s' with %d queries prepared.", workload.description, len(workload.queries))

    connector = PostgresDatabaseConnector(database_name, autocommit=True)
    # connector.enable_simulation()
    if args.analyze:
        connector.create_statistics()

    try:
        baseline_cost, indexed_cost, baseline_costs, indexed_costs = evaluate_cost(
            workload, indexes, connector, args.cost_mode, args.per_query
        )
    finally:
        connector.close()

    print("=== Index Configuration Evaluation ===")
    print(f"Benchmark:           {args.benchmark}")
    print(f"Database:            {database_name}")
    print(f"Workload:            {workload.description}")
    print(f"Indexes evaluated:   {len(indexes)}")
    for idx_obj in indexes:
        size_mb = (idx_obj.estimated_size or 0) / (1024 * 1024)
        print(f"  - {idx_obj}  (~{size_mb:.2f} MB)")

    print("\nCost summary (lower is better):")
    print(f"  Baseline cost:     {baseline_cost:.2f}")
    print(f"  Indexed cost:      {indexed_cost:.2f}")
    if baseline_cost > 0:
        improvement = (baseline_cost - indexed_cost) / baseline_cost * 100
        print(f"  Improvement:       {improvement:.2f}%")

    if args.per_query:
        print("\nPer-query costs:")
        for query, base_cost, idx_cost in zip(workload.queries, baseline_costs, indexed_costs):
            delta = base_cost - idx_cost
            print(f"  Q{query.nr:02d}: baseline={base_cost:.2f}, indexed={idx_cost:.2f}, delta={delta:.2f}")


if __name__ == "__main__":
    main()

