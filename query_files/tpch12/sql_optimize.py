import sqlglot
from sqlglot.optimizer import optimize, qualify_columns, simplify
import psycopg2
from psycopg2.extras import RealDictCursor, quote_ident
import argparse
import sys
import os

class SQLOptimizer:
    def __init__(self, db_config=None):
        self.db_config = db_config or {
            'host': '/tmp',
            'database': 'your_database',
            'port': 51204
        }
        self.schema_cache = {}
    
    def get_connection(self):
        """Create database connection"""
        try:
            return psycopg2.connect(**self.db_config)
        except psycopg2.Error as e:
            raise Exception(f"Database connection failed: {str(e)}")
    
    def get_table_schema(self, table_name):
        """Get column information for a table"""
        if table_name in self.schema_cache:
            return self.schema_cache[table_name]
        
        query = """
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = %s 
            ORDER BY ordinal_position
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (table_name,))
                    columns = {row['column_name']: row['data_type'] for row in cur.fetchall()}
            
            self.schema_cache[table_name] = columns
            return columns
        except Exception as e:
            raise Exception(f"Failed to get schema for table '{table_name}': {str(e)}")
    
    def extract_tables_from_sql(self, sql):
        """Extract table names from SQL query"""
        try:
            parsed = sqlglot.parse_one(sql)
            tables = set()
            
            for table in parsed.find_all(sqlglot.expressions.Table):
                tables.add(table.name)
            
            return list(tables)
        except Exception as e:
            raise Exception(f"Failed to parse SQL: {str(e)}")
    
    def build_schema_dict(self, tables):
        """Build schema dictionary for SQLGlot"""
        schema = {}
        for table in tables:
            schema[table] = self.get_table_schema(table)
        return schema
    
    def optimize_sql(self, sql):
        """Main method to optimize SQL - returns optimized SQL string"""
        try:
            # Skip empty lines and comments
            if not sql.strip() or sql.strip().startswith('--'):
                return sql
            
            # Extract tables from SQL
            tables = self.extract_tables_from_sql(sql)
            if not tables:
                return sql  # Return original if no tables found
            
            # Get schema information
            schema = self.build_schema_dict(tables)
            
            # Let SQLGlot handle all optimization including column qualification
            # optimized_ast = optimize(sqlglot.parse_one(sql), schema=schema, rules=[qualify_columns])
            optimized_ast = qualify_columns.qualify_columns(sqlglot.parse_one(sql), schema=schema)
            optimized_ast = simplify.simplify(optimized_ast)
            
            for node in optimized_ast.find_all(sqlglot.exp.Identifier):
                node.set("quoted", False)
            for select in optimized_ast.find_all(sqlglot.exp.Alias):  
                select.replace(select.this)
            # Convert back to SQL
            optimized_sql = optimized_ast.sql(identify=False, dialect='postgres', normalize_functions=False)
            
            return optimized_sql
            
        except sqlglot.errors.OptimizeError as e:
            # If SQLGlot can't resolve ambiguity, provide better error message
            error_msg = str(e)
            if "could not be resolved" in error_msg:
                print(f"Warning: Column ambiguity detected. SQLGlot cannot resolve: {error_msg}", file=sys.stderr)
                print("Returning original SQL. Consider adding table qualifications manually.", file=sys.stderr)
            return sql
        except Exception as e:
            # Return original SQL if optimization fails
            print(f"Warning: Optimization failed, returning original SQL. Error: {str(e)}", file=sys.stderr)
            return sql

def read_sql_file(file_path):
    """Read SQL file and return list of SQL statements"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sql_statements = f.readlines()
        if sql_statements[-1] == '':
            sql_statements.pop()
        return sql_statements
    except Exception as e:
        raise Exception(f"Failed to read SQL file: {str(e)}")

def write_sql_file(file_path, sql_statements):
    """Write SQL statements to file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for sql in sql_statements:
                f.write(sql + ';\n')
        return True
    except Exception as e:
        raise Exception(f"Failed to write SQL file: {str(e)}")

def process_sql_file(input_file, output_file, db_config):
    """Process a file containing multiple SQL statements"""
    # Read SQL statements from file
    sql_statements = read_sql_file(input_file)
    print(f"Found {len(sql_statements)} SQL statements in {input_file}")
    
    # Initialize optimizer
    optimizer = SQLOptimizer(db_config)
    
    # Optimize each SQL statement
    optimized_statements = []
    for i, sql in enumerate(sql_statements, 1):
        print(f"Optimizing statement {i}/{len(sql_statements)}...")
        optimized_sql = optimizer.optimize_sql(sql)
        optimized_statements.append(optimized_sql)
    
    # Write optimized SQL to output file
    write_sql_file(output_file, optimized_statements)
    print(f"Optimized SQL written to {output_file}")
    
    return True
        
    # except Exception as e:
    #     print(f"Error processing file: {str(e)}", file=sys.stderr)
    #     return False

def main():
    parser = argparse.ArgumentParser(description='Optimize SQL queries using PostgreSQL schema information')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--sql', help='Single SQL query to optimize')
    input_group.add_argument('--file', help='Input file containing SQL queries (one per line)')
    
    # Output options
    parser.add_argument('--output', '-o', help='Output file (required for --file, optional for --sql)')
    
    # Database connection options
    parser.add_argument('--host', default='/tmp', help='PostgreSQL host')
    parser.add_argument('--database', required=True, help='PostgreSQL database name')
    parser.add_argument('--port', default=51204, type=int, help='PostgreSQL port')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.file and not args.output:
        parser.error("--output is required when using --file")
    
    # Build database configuration
    db_config = {
        'host': args.host,
        'database': args.database,
        'port': args.port
    }
    
    try:
        if args.sql:
            # Single SQL mode
            optimizer = SQLOptimizer(db_config)
            optimized_sql = optimizer.optimize_sql(args.sql)
            
            if args.output:
                # Write to file
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(optimized_sql)
                print(f"Optimized SQL written to {args.output}")
            else:
                # Print to stdout
                print(optimized_sql)
                
        else:
            # File mode
            success = process_sql_file(args.file, args.output, db_config)
            if not success:
                sys.exit(1)
                
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

# Function to use directly in other Python code
def optimize_sql_string(sql_string, db_config=None):
    """
    Optimize a SQL string and return the optimized version.
    
    Args:
        sql_string (str): The SQL query to optimize
        db_config (dict): PostgreSQL connection configuration
    
    Returns:
        str: Optimized SQL query
    """
    optimizer = SQLOptimizer(db_config)
    return optimizer.optimize_sql(sql_string)

def optimize_sql_file(input_file, output_file, db_config=None):
    """
    Optimize all SQL statements in a file and write to output file.
    
    Args:
        input_file (str): Path to input SQL file
        output_file (str): Path to output SQL file
        db_config (dict): PostgreSQL connection configuration
    
    Returns:
        bool: True if successful, False otherwise
    """
    return process_sql_file(input_file, output_file, db_config)

def test_quote_identifiers():
    # Parse a SQL statement
    expression = sqlglot.parse_one("SELECT column_name FROM table_name")
    expression = optimize(expression) 
    for node in expression.find_all(sqlglot.exp.Identifier):
        node.set("quoted", False)
    # Generate SQL without quoting identifiers
    sql = expression.sql(identify=False, dialect='postgres')
    print(sql)

def test_kw_cases():
    sql = "select nation.n_name from region"  
    parsed = sqlglot.parse_one(sql, dialect="postgres")  
      
    # Make sure normalize_functions is set on the .sql() call  
    output = parsed.sql(  
        dialect="postgres",  
        normalize_functions=False  # This should preserve lowercase keywords  
    )
    print(output)

if __name__ == "__main__":
    # test_kw_cases()
    # test_quote_identifiers()
    main()
