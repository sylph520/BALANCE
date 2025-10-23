import psycopg2
import pickle
import numpy as np
import json
import os
import argparse
from  src.plan_encoding.meta_info  import get_tables

# --- User Configuration ---
# IMPORTANT: Replace with your PostgreSQL database credentials and TPCDS database name
DB_USER = "sclai"  # Your PostgreSQL username
DB_PASSWORD = ""  # Your PostgreSQL password
DB_HOST = "/tmp"  # Your PostgreSQL host
DB_PORT = "51204"  # Your PostgreSQL port

# Number of bins for discretization (matches parameters.box_num in the project)
NUM_BINS = 10

# Output pickle file name
# --- End User Configuration ---

def generate_box_lines(db_name, num_bins, format_type):
    try:
        conn = psycopg2.connect(dbname=db_name, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        conn.autocommit = True # Ensure autocommit for DDL like CREATE EXTENSION
        cur = conn.cursor()
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        print("Please ensure your PostgreSQL database is running and credentials are correct.")
        return None

    box_lines_data = {}

    # Get all tables in the public schema
    cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
    tables = [row[0] for row in cur.fetchall()]

    # Filter for known TPCDS tables (based on previous debug output and common TPCDS tables)
    # tpcds_tables = [
    #     'inventory', 'catalog_sales', 'web_sales', 'date_dim', 'promotion',
    #     'store_sales', 'store_returns', 'customer', 'item', 'call_center',
    #     'web_returns', 'store', 'web_site', 'warehouse', 'reason', 'income_band',
    #     'time_dim', 'household_demographics', 'customer_address', 'customer_demographics',
    #     'web_page', 'catalog_returns', 'catalog_page',
    #     'ship_mode', 'state_province', 'zip_code', 'country'
    #     # Excluded history tables for simplicity, add if needed
    # ]
    bm_tables = get_tables(conn)
    tables = [t for t in tables if t in bm_tables]


    for table_name in tables:
        box_lines_data[table_name] = {}
        print(f"Processing table: {table_name}")

        # Get columns and their types for the current table
        cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';")
        columns_info = cur.fetchall()

        for col_name, data_type in columns_info:
            full_col_name = f"{table_name}.{col_name}"

            if format_type == 'new':
                # New format: min/max for numeric, bins for string
                if data_type in ['integer', 'numeric', 'double precision', 'real', 'smallint', 'bigint', 'decimal']:
                    try:
                        cur.execute(f"SELECT MIN({col_name}), MAX({col_name}) FROM {table_name} WHERE {col_name} IS NOT NULL;")
                        min_val, max_val = cur.fetchone()

                        if min_val is not None and max_val is not None:
                            box_lines_data[table_name][col_name] = {'min': float(min_val), 'max': float(max_val), 'type': 'numeric'}
                            print(f"  - Collected min/max for numeric column {full_col_name}: min={min_val}, max={max_val}")
                        else:
                            print(f"  - No data for numeric column {full_col_name}, skipping min/max collection.")
                    except Exception as e:
                        print(f"  - Could not collect min/max for numeric column {full_col_name}: {e}")

                elif data_type in ['character varying', 'text', 'character', 'bpchar']:
                    try:
                        cur.execute(f"SELECT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL ORDER BY {col_name} LIMIT 100000;")
                        data = [str(row[0]) for row in cur.fetchall()]

                        if len(data) > 0:
                            indices = np.linspace(0, len(data) - 1, num_bins + 1, dtype=int)[1:-1]
                            bins = [data[i] for i in indices]
                            box_lines_data[table_name][col_name] = {'bins': sorted(list(set(bins))), 'type': 'string'}
                            print(f"  - Generated {len(box_lines_data[table_name][col_name]['bins'])} bins for string column {full_col_name}")
                        else:
                            print(f"  - No data for string column {full_col_name}, skipping bin generation.")
                    except Exception as e:
                        print(f"  - Could not generate bins for string column {full_col_name}: {e}")

            elif format_type == 'old':
                # Old format: bins for both numeric and string
                if data_type in ['integer', 'numeric', 'double precision', 'real', 'smallint', 'bigint', 'decimal']:
                    try:
                        cur.execute(f"SELECT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL ORDER BY {col_name} LIMIT 100000;")
                        data = [float(row[0]) for row in cur.fetchall()]
                        if len(data) > 0:
                            bins = np.percentile(data, np.linspace(0, 100, num_bins + 1)[1:-1])
                            box_lines_data[table_name][col_name] = sorted(list(set(bins)))
                            print(f"  - Generated {len(box_lines_data[table_name][col_name])} bins for numeric column {full_col_name} (old format).")
                        else:
                            print(f"  - No data for numeric column {full_col_name}, skipping bin generation (old format).")
                    except Exception as e:
                        print(f"  - Could not generate bins for numeric column {full_col_name} (old format): {e}")

                elif data_type in ['character varying', 'text', 'character', 'bpchar']:
                    try:
                        cur.execute(f"SELECT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL ORDER BY {col_name} LIMIT 100000;")
                        data = [str(row[0]) for row in cur.fetchall()]
                        if len(data) > 0:
                            indices = np.linspace(0, len(data) - 1, num_bins + 1, dtype=int)[1:-1]
                            bins = [data[i] for i in indices]
                            box_lines_data[table_name][col_name] = sorted(list(set(bins)))
                            print(f"  - Generated {len(box_lines_data[table_name][col_name])} bins for string column {full_col_name} (old format).")
                        else:
                            print(f"  - No data for string column {full_col_name}, skipping bin generation (old format).")
                    except Exception as e:
                        print(f"  - Could not generate bins for string column {full_col_name} (old format): {e}")

    cur.close()
    conn.close()
    return box_lines_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate box_line.pickle for TPCDS.")
    # parser.add_argument('--bm', type=str,  default='tpcds')
    parser.add_argument('--bm', type=str,  default='tpch')
    # parser.add_argument('--format', type=str, default='new', choices=['old', 'new'],
    parser.add_argument('--format', type=str, default='old', choices=['old', 'new'],
                        help="Format of the generated box_line.pickle: 'old' (list of bins) or 'new' (min/max for numeric, bins for string).")
    args = parser.parse_args()

    if args.bm.lower() in ['tpch']:
        DB_NAME = "indexselection_tpch___1"  # The name of your TPCDS database
    elif args.bm.lower() in ['tpcds']:
        DB_NAME = "indexselection_tpcds___10"  # The name of your TPCDS database
    OUTPUT_PICKLE_FILE = f"{args.bm.lower()}_box_line_{args.format}.pickle"
    print(f"Starting {OUTPUT_PICKLE_FILE} generation script...")

    bm_box_lines = generate_box_lines(DB_NAME, NUM_BINS, args.format)

    if bm_box_lines is not None:
        with open(OUTPUT_PICKLE_FILE, 'wb') as f:
            pickle.dump(bm_box_lines, f)
        print(f"'{OUTPUT_PICKLE_FILE}' generated successfully in '{args.format}' format.")
    else:
        print(f"Failed to generate '{OUTPUT_PICKLE_FILE}'. Please check error messages above.")
