import os
import psycopg2
import sqlparse
import random
import argparse
import datetime

# --- Configuration ---
DB_NAME = "indexselection_tpch___1"
DB_PORT = 51204
DB_HOST = "/tmp"
DB_USER = "postgres"
DB_PASSWORD = "" # Assuming no password for local connection

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIR = SCRIPT_DIR
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "generated_queries")

# --- Database Connection ---
def get_db_connection():
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error: Could not connect to the database: {e}")
        return None

# --- Value Generation ---
def get_random_value_for_equality(table, column, conn):
    with conn.cursor() as cur:
        try:
            cur.execute(f"SELECT {column} FROM {table} ORDER BY RANDOM() LIMIT 1")
            return cur.fetchone()[0]
        except (psycopg2.Error, TypeError) as e:
            print(f"  - DB Error [get_random_value_for_equality]: {e}")
            return None

def get_random_value_for_range(table, column, conn):
    with conn.cursor() as cur:
        try:
            cur.execute(f"SELECT MIN({column}), MAX({column}) FROM {table}")
            min_val, max_val = cur.fetchone()
            if min_val is None or max_val is None: return None

            if isinstance(min_val, datetime.date):
                days = (max_val - min_val).days
                return min_val + datetime.timedelta(days=random.randint(0, days))
            elif isinstance(min_val, (int, float)):
                return random.uniform(float(min_val), float(max_val))
            else:
                return random.choice([min_val, max_val])
        except (psycopg2.Error, TypeError) as e:
            print(f"  - DB Error [get_random_value_for_range]: {e}")
            return None

# --- SQL Parsing Logic ---
def get_tables_from_query(parsed):
    tables = {}
    from_seen = False
    for token in parsed.tokens:
        if token.is_keyword and token.normalized == 'FROM': from_seen = True
        elif from_seen:
            if isinstance(token, sqlparse.sql.IdentifierList):
                for identifier in token.get_identifiers():
                    tables[identifier.get_alias() or identifier.get_real_name()] = identifier.get_real_name()
            elif isinstance(token, sqlparse.sql.Identifier):
                tables[token.get_alias() or token.get_real_name()] = token.get_real_name()
            if token.is_keyword and token.normalized in ('WHERE', 'GROUP', 'ORDER', 'LIMIT'): break
    return tables

def generate_variant(original_sql, tables, conn):
    parsed = sqlparse.parse(original_sql)[0]
    where_clause = next((t for t in parsed.tokens if isinstance(t, sqlparse.sql.Where)), None)
    if not where_clause: return original_sql

    # We work on a copy of the sublists to avoid issues while modifying
    for comparison in list(where_clause.get_sublists()):
        if not isinstance(comparison, sqlparse.sql.Comparison): continue

        right_side = comparison.right
        column_identifier = comparison.left
        
        op_token = None
        for t in comparison.tokens:
            if t.ttype in (sqlparse.tokens.Operator, sqlparse.tokens.Comparison):
                op_token = t
                break

        if not (right_side and column_identifier and op_token): continue

        is_simple_literal = right_side.ttype in sqlparse.tokens.Literal
        is_typed_literal = isinstance(right_side, sqlparse.sql.TypedLiteral)

        if not (is_simple_literal or is_typed_literal): continue

        full_column_name = str(column_identifier)
        parts = full_column_name.split('.')
        column_name = parts[-1]
        table_alias = parts[0] if len(parts) > 1 else None
        table_name = tables.get(table_alias) if table_alias else list(tables.values())[0] if len(tables) == 1 else None

        if not table_name: continue

        operator = op_token.normalized
        new_value = None

        if operator == '=':
            new_value = get_random_value_for_equality(table_name, column_name, conn)
        elif operator in ('>', '<', '>=', '<='):
            new_value = get_random_value_for_range(table_name, column_name, conn)

        if new_value is not None:
            # --- FINAL FIX: Handle simple and complex literals separately ---
            if isinstance(new_value, datetime.date):
                # This will always be a complex literal
                new_tokens = [sqlparse.sql.Token(sqlparse.tokens.Keyword, 'date'),
                              sqlparse.sql.Token(sqlparse.tokens.Literal.String.Single, f"'{new_value.strftime('%Y-%m-%d')}'")]
                right_side.tokens = new_tokens
            elif isinstance(right_side, sqlparse.sql.TypedLiteral):
                 # Handle other typed literals if they exist
                 pass # No other typed literals to handle for now
            else:
                # This is a simple literal token, so modify its value
                if isinstance(new_value, str):
                    right_side.value = f"'{new_value}'"
                elif isinstance(new_value, float):
                    right_side.value = f"{new_value:.2f}"
                else:
                    right_side.value = str(new_value)

    return str(parsed)

# --- Main Logic ---
def process_queries(n_variants):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    conn = get_db_connection()
    if not conn: return

    print(f"Looking for queries in: {os.path.abspath(SOURCE_DIR)}")
    
    for filename in os.listdir(SOURCE_DIR):
        filepath = os.path.join(SOURCE_DIR, filename)
        if not os.path.isfile(filepath) or not filename.endswith(('.txt', '.sql')):
            continue

        print(f"Processing: {filename}")
        with open(filepath, 'r') as f:
            original_sql = f.read().strip()
        if not original_sql: continue

        tables = get_tables_from_query(sqlparse.parse(original_sql)[0])

        if not tables:
            print(f"  - Could not determine tables for {filename}. Skipping.")
            continue

        variants = [original_sql]
        for _ in range(n_variants - 1):
            # We must re-parse the original sql each time to avoid modifying it in place
            new_variant = generate_variant(original_sql, tables, conn)
            variants.append(new_variant.strip())
        
        output_content = "\n".join(variants)
        output_filename = os.path.join(OUTPUT_DIR, filename)
        
        with open(output_filename, 'w') as f:
            f.write(output_content)
        print(f"  -> Generated {len(variants)} variants into: {output_filename}")

    conn.close()
    print("\nProcessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate variants of SQL queries.")
    parser.add_argument("--n", type=int, default=2, help="Total number of queries (original + variants) to generate.")
    args = parser.parse_args()
    
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    SOURCE_DIR = SCRIPT_DIR
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "generated_queries")

    process_queries(args.n)
