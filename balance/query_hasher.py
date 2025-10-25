import re
import hashlib
import argparse
from pglast.stream import RawStream
from balance.sql_optimize import optimize_sql_string

def query_to_hash_text(query: str, unify_similar_ops: bool = False, dbname='') -> tuple:
    """
    Convert SQL query to normalized template hash using text operations.
    """
    if not query or not query.strip():
        raise ValueError("query_to_hash_text: empty incoming query")

    # __import__('ipdb').set_trace()
    query = optimize_sql_string(query, {"database": dbname, "host":"/tmp", "port": 51204})

    # Normalize: lowercase, single spaces
    normalized = re.sub(r'\s+', ' ', query.strip()).lower()

    # Optionally unify all comparison operators to a single token
    if unify_similar_ops:
        normalized = re.sub(r'<=|>=|<|>|=', ' CMP ', normalized)

    # Replace all parameter types with ?
    patterns = [
        (r"cast\s*\(\s*'[^']*'\s*as\s*\w+\s*\)", "?"),  # CAST('value' AS type) - entire expression
        (r"'[^']*'", "?"),           # Strings
        (r'"[^"]*"', "?"),           # Identifiers
        (r'\b\d+\.?\d*\b', "?"),     # Numbers
        (r'\b(true|false|null)\b', "?"),  # Booleans
        (r'\$\d+', "?"),             # PG params
    ]

    for pattern, replacement in patterns:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    # Clean up and hash
    normalized = re.sub(r'\?\s*\?', "?", normalized).strip()
    return hashlib.md5(normalized.encode()).hexdigest(), normalized

def query_to_hash_ast(query: str, unify_similar_ops, dbname) -> str:
    """
    Convert SQL query to normalized template hash using AST parsing.
    """
    try:
        import pglast
    except ImportError:
        print("Warning: pglast not available, falling back to text-based classification")
        return query_to_hash_text(query)

    try:
        # Parse the query into AST
        parsed = pglast.parse_sql(query)

        def normalize_node(node):
            """Recursively normalize AST nodes by replacing literals with placeholders"""
            if isinstance(node, dict):
                # Replace literals with placeholders
                if node.get('A_Const'):
                    return '?'

                # Recursively process other nodes
                new_node = {}
                for key, value in node.items():
                    if isinstance(value, list):
                        new_node[key] = [normalize_node(item) for item in value]
                    elif isinstance(value, dict):
                        new_node[key] = normalize_node(value)
                    else:
                        new_node[key] = value
                return new_node

            elif isinstance(node, list):
                return [normalize_node(item) for item in node]
            else:
                return node

        # Normalize the AST
        normalized_ast = normalize_node(parsed)

        # Convert back to SQL (this gives us the normalized query)
        normalized_sql = RawStream()(normalized_ast)

        # Generate hash
        return hashlib.md5(normalized_sql.encode()).hexdigest(), normalized_ast

    except Exception as e:
        print(f"AST parsing failed for query, falling back to text: {e}")
        return query_to_hash_text(query)

def query_to_hash(query: str, use_ast: bool = False, unify_similar_ops: bool = False, dbname='') -> str:
    """
    Convert SQL query to normalized template hash.

    Args:
        query: SQL query string
        use_ast: If True, use AST parsing; if False, use text operations
        unify_similar_ops: If True, treat all comparison operators as the same
    """
    if use_ast:
        return query_to_hash_ast(query, unify_similar_ops=unify_similar_ops, dbname=dbname)
    else:
        return query_to_hash_text(query, unify_similar_ops=unify_similar_ops, dbname=dbname)

def process_sql_file(filename: str, use_ast: bool = False, unify_similar_ops: bool = False, dbname='') -> dict:
    """
    Process a SQL file and count query templates.

    Args:
        filename: Path to SQL file (one query per line)
        use_ast: Whether to use AST parsing
        unify_similar_ops: If True, treat all comparison operators as the same

    Returns:
        Dictionary with template counts and examples
    """
    templates = {}

    try:
        with open(filename, 'r') as file:
            for line_num, line in enumerate(file, 1):
                query = line.strip()

                # Skip empty lines and comments
                if not query or query.startswith('--'):
                    continue

                template_hash, _ = query_to_hash(query, use_ast, unify_similar_ops, dbname=dbname)
                if template_hash:
                    if template_hash not in templates:
                        templates[template_hash] = {
                            'count': 0,
                            'examples': []
                        }

                    templates[template_hash]['count'] += 1
                    # Keep first 2 examples
                    if len(templates[template_hash]['examples']) < 2:
                        templates[template_hash]['examples'].append(query)

        print(f"Processed {line_num} lines from {filename}")
        return templates

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return {}

def generate_sample_file(filename: str = "sample_queries.sql", num_queries: int = 20):
    """Generate a sample SQL file for testing."""
    sample_queries = [
        "SELECT * FROM users WHERE id = 1",
        "SELECT * FROM users WHERE id = 42",
        "SELECT name, email FROM users WHERE age > 25",
        "SELECT name, email FROM users WHERE age > 30",
        "INSERT INTO products (name, price) VALUES ('Laptop', 999.99)",
        "INSERT INTO products (name, price) VALUES ('Phone', 499.99)",
        "UPDATE orders SET status = 'shipped' WHERE order_id = 123",
        "UPDATE orders SET status = 'delivered' WHERE order_id = 456",
        "SELECT * FROM logs WHERE level = 'ERROR' AND timestamp > '2023-01-01'",
        "SELECT * FROM logs WHERE level = 'INFO' AND timestamp > '2023-02-01'",
    ]

    with open(filename, 'w') as f:
        f.write("-- Sample SQL queries\n")
        for i in range(num_queries):
            query = sample_queries[i % len(sample_queries)]
            f.write(query + "\n")

    print(f"Generated sample file: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Convert SQL queries to template hashes')
    parser.add_argument('filename', nargs='?', help='SQL file to process')
    parser.add_argument('--query', help='Single query to convert')
    parser.add_argument('--dbname', help='dbname')
    parser.add_argument('--ast', action='store_true', help='Use AST parsing instead of text operations')
    parser.add_argument('--unify-ops', action='store_true', help='Unify comparison operators like <, > into a single token')
    parser.add_argument('--generate-sample', type=int, help='Generate sample file with N queries')

    args = parser.parse_args()

    if args.generate_sample:
        generate_sample_file(num_queries=args.generate_sample)
        return

    method = "AST" if args.ast else "text"

    if args.query:
        # Process single query
        template_hash = query_to_hash(args.query, args.ast, args.unify_ops, args.dbname)
        print(f"Method: {method}")
        print(f"Query: {args.query}")
        print(f"Template Hash: {template_hash}")

    elif args.filename:
        # Process file
        templates = process_sql_file(args.filename, args.ast, args.unify_ops)
        print(f"\nMethod: {method}")
        print(f"Found {len(templates)} unique query templates:")

        for hash_val, info in templates.items():
            print(f"\n{hash_val}:")
            print(f"  Count: {info['count']} queries")
            print(f"  Examples:")
            for example in info['examples']:
                print(f"    - {example}")

    else:
        print("Usage:")
        print("  python query_hasher.py --query \"SELECT * FROM users WHERE id = 1\"")
        print("  python query_hasher.py --query \"SELECT * FROM users WHERE id = 1\" --ast")
        print("  python query_hasher.py queries.sql")
        print("  python query_hasher.py queries.sql --ast")
        print("  python query_hasher.py --generate-sample 50")

# Test function
def test_both_methods():
    """Test both text and AST methods on sample queries."""
    test_queries = [
        "SELECT * FROM users WHERE id = 1",
        "SELECT * FROM users WHERE id = 42",
        "INSERT INTO products (name, price) VALUES ('Laptop', 999.99)",
        "INSERT INTO products (name, price) VALUES ('Phone', 499.99)",
        "UPDATE orders SET status = 'shipped' WHERE order_id = 123",
        "UPDATE orders SET status = 'delivered' WHERE order_id = 456",
    ]

    print("Comparing Text vs AST methods:")
    print("=" * 60)

    for query in test_queries:
        text_hash, _ = query_to_hash_text(query)
        ast_hash = query_to_hash_ast(query)

        print(f"Query: {query}")
        print(f"  Text: {text_hash}")
        print(f"  AST:  {ast_hash}")
        print(f"  Match: {text_hash == ast_hash}")
        print()

if __name__ == "__main__":
    main()

    # Uncomment to run comparison test
    # test_both_methods()
