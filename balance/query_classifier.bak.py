import re
import hashlib
from typing import Tuple, Dict, List
import argparse
import sys

def classify_query_text(query: str) -> Tuple[str, str]:
    """
    Classify query using text-based normalization.
    Returns: (template_hash, normalized_query)
    """
    # Remove extra whitespace and convert to lowercase for consistency
    normalized = re.sub(r'\s+', ' ', query.strip()).lower()
    
    # Skip empty queries
    if not normalized:
        return None, None
    
    # Replace various parameter patterns with placeholders
    # 1. Single quoted strings
    normalized = re.sub(r"'[^']*'", "?", normalized)
    
    # 2. Double quoted strings (for identifiers)
    normalized = re.sub(r'"[^"]*"', "?", normalized)
    
    # 3. Numeric values (integers and decimals)
    normalized = re.sub(r'\b\d+\.?\d*\b', "?", normalized)
    
    # 4. Boolean values
    normalized = re.sub(r'\b(true|false|null)\b', "?", normalized, flags=re.IGNORECASE)
    
    # 5. PostgreSQL style parameters ($1, $2, etc.)
    normalized = re.sub(r'\$\d+', "?", normalized)
    
    # 6. JDBC style parameters (?)
    # Note: We handle this carefully to avoid replacing existing placeholders
    normalized = re.sub(r'\b\?\b', "?", normalized)
    
    # Final cleanup of multiple consecutive placeholders or spaces
    normalized = re.sub(r'\?\s*\?', "?", normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Generate hash for the template
    template_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    return template_hash, normalized

def classify_query_ast(query: str) -> Tuple[str, str]:
    """
    Classify query using AST parsing with pglast.
    Returns: (template_hash, normalized_query)
    """
    try:
        import pglast
    except ImportError:
        print("Warning: pglast not available, falling back to text-based classification")
        return classify_query_text(query)
    
    try:
        # Parse the query into AST
        parsed = pglast.parse_sql(query)
        
        def normalize_node(node):
            """Recursively normalize AST nodes by replacing literals with placeholders"""
            if isinstance(node, dict):
                # Replace literals with placeholders
                if node.get('A_Const'):
                    const = node['A_Const']
                    if 'val' in const:
                        return '?'
                elif node.get('ColumnRef'):
                    # Keep column references as is
                    fields = node['ColumnRef']['fields']
                    field_names = []
                    for field in fields:
                        if 'String' in field:
                            field_names.append(field['String']['str'])
                        elif 'A_Star' in field:
                            field_names.append('*')
                    return '.'.join(field_names)
                
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
        normalized_sql = pglast.print_sql(normalized_ast)
        
        # Generate hash for the template
        template_hash = hashlib.md5(normalized_sql.encode()).hexdigest()[:16]
        
        return template_hash, normalized_sql
    
    except Exception as e:
        print(f"AST parsing failed for query: {query[:100]}... Error: {e}")
        # Fall back to text-based classification
        return classify_query_text(query)

def classify_query(query: str, use_ast: bool = False) -> Tuple[str, str]:
    """
    Main function to classify a query into a template.
    
    Args:
        query: The SQL query string
        use_ast: Whether to use AST parsing (more accurate) or text-based
    
    Returns:
        Tuple of (template_hash, normalized_query)
    """
    if use_ast:
        return classify_query_ast(query)
    else:
        return classify_query_text(query)

def process_query_stream_from_file(filename: str, use_ast: bool = False, max_queries: int = None) -> Dict[str, Dict]:
    """
    Process queries from a file where each line represents a query.
    
    Args:
        filename: Path to the SQL file
        use_ast: Whether to use AST parsing
        max_queries: Maximum number of queries to process (None for all)
    
    Returns:
        Dictionary mapping template_hash to template info and query examples
    """
    templates = {}
    query_count = 0
    skipped_queries = 0
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                # Skip empty lines and comments
                stripped_line = line.strip()
                if not stripped_line or stripped_line.startswith('--'):
                    continue
                
                # Check if we've reached the maximum number of queries
                if max_queries and query_count >= max_queries:
                    print("Processed {} queries, stopping...".format(max_queries))
                    break
                
                query_count += 1
                
                try:
                    template_hash, normalized = classify_query(stripped_line, use_ast)
                    
                    # Skip if classification returned None (empty query after normalization)
                    if template_hash is None:
                        skipped_queries += 1
                        continue
                    
                    if template_hash not in templates:
                        templates[template_hash] = {
                            'template': normalized,
                            'count': 0,
                            'examples': [],
                            'line_numbers': []
                        }
                    
                    templates[template_hash]['count'] += 1
                    templates[template_hash]['line_numbers'].append(line_num)
                    
                    # Keep only first 3 examples to save memory
                    if len(templates[template_hash]['examples']) < 3:
                        templates[template_hash]['examples'].append(
                            stripped_line[:200] + '...' if len(stripped_line) > 200 else stripped_line
                        )
                
                except Exception as e:
                    print("Error processing line {}: {}".format(line_num, e))
                    skipped_queries += 1
                    continue
    
    except FileNotFoundError:
        print("Error: File '{}' not found.".format(filename))
        return {}
    except Exception as e:
        print("Error reading file: {}".format(e))
        return {}
    
    print("Processed {} queries from file ({} skipped)".format(query_count, skipped_queries))
    return templates

def generate_sample_sql_file(filename: str, num_queries: int = 50):
    """
    Generate a sample SQL file for testing purposes.
    """
    sample_queries = [
        "SELECT * FROM users WHERE id = 1",
        "SELECT * FROM users WHERE id = 42",
        "SELECT name, email FROM users WHERE age > 25",
        "SELECT name, email FROM users WHERE age > 30",
        "INSERT INTO products (name, price) VALUES ('Laptop', 999.99)",
        "INSERT INTO products (name, price) VALUES ('Phone', 499.99)",
        "UPDATE orders SET status = 'shipped' WHERE order_id = 123",
        "UPDATE orders SET status = 'delivered' WHERE order_id = 456",
        "DELETE FROM sessions WHERE expires_at < '2023-01-01'",
        "DELETE FROM sessions WHERE expires_at < '2023-06-01'",
        "SELECT COUNT(*) FROM orders WHERE user_id = 100 AND status = 'completed'",
        "SELECT COUNT(*) FROM orders WHERE user_id = 200 AND status = 'pending'",
        "SELECT u.username, o.total FROM users u JOIN orders o ON u.id = o.user_id WHERE u.country = 'US'",
        "SELECT u.username, o.total FROM users u JOIN orders o ON u.id = o.user_id WHERE u.country = 'UK'",
        "UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 5",
        "UPDATE inventory SET quantity = quantity - 3 WHERE product_id = 8",
        "SELECT * FROM logs WHERE level = 'ERROR' AND timestamp > '2023-01-01'",
        "SELECT * FROM logs WHERE level = 'INFO' AND timestamp > '2023-02-01'",
    ]
    
    with open(filename, 'w', encoding='utf-8') as f:
        # Write some comments
        f.write("-- Sample SQL queries for testing query classification\n")
        f.write("-- This file contains various query templates with different parameters\n\n")
        
        # Repeat sample queries to fill the file
        for i in range(num_queries):
            query = sample_queries[i % len(sample_queries)]
            # Add some variations
            if i % 5 == 0:
                query = query.upper()  # Some in uppercase
            elif i % 7 == 0:
                query = "   " + query + "   "  # Some with extra spaces
            
            f.write(query + "\n")
    
    print("Generated sample SQL file: {} with {} queries".format(filename, num_queries))

def print_templates(templates: Dict[str, Dict], output_format: str = "detailed"):
    """
    Print the classified templates in various formats.
    """
    if not templates:
        print("No templates found.")
        return
    
    print("\nFound {} unique query templates".format(len(templates)))
    print("=" * 80)
    
    if output_format == "summary":
        # Simple summary format
        for i, (template_hash, info) in enumerate(templates.items(), 1):
            print("{}. Template: {}".format(i, info['template']))
            print("   Count: {} queries".format(info['count']))
            line_nums_preview = info['line_numbers'][:5]
            if len(info['line_numbers']) > 5:
                print("   Lines: {}...".format(line_nums_preview))
            else:
                print("   Lines: {}".format(line_nums_preview))
            print()
    
    elif output_format == "detailed":
        # Detailed format with examples
        for i, (template_hash, info) in enumerate(templates.items(), 1):
            print("Template {}:".format(i))
            print("  Hash: {}".format(template_hash))
            print("  Pattern: {}".format(info['template']))
            print("  Frequency: {} occurrences".format(info['count']))
            line_nums_preview = info['line_numbers'][:10]
            if len(info['line_numbers']) > 10:
                print("  Line numbers: {}...".format(line_nums_preview))
            else:
                print("  Line numbers: {}".format(line_nums_preview))
            print("  Examples:")
            for example in info['examples']:
                print("    - {}".format(example))
            print("-" * 60)
    
    elif output_format == "csv":
        # CSV-like format for easy processing
        print("template_hash,pattern,count,example_line_numbers")
        for template_hash, info in templates.items():
            line_nums = ";".join(map(str, info['line_numbers'][:5]))
            # Fixed the f-string syntax issue
            template_escaped = info['template'].replace('"', '""')  # Escape double quotes for CSV
            print('{},"{}",{},"{}"'.format(template_hash, template_escaped, info['count'], line_nums))

def main():
    parser = argparse.ArgumentParser(description='Classify SQL queries from a file into templates')
    parser.add_argument('filename', nargs='?', help='SQL file to process (each line = one query)')
    parser.add_argument('--use-ast', action='store_true', help='Use AST parsing (requires pglast)')
    parser.add_argument('--max-queries', type=int, help='Maximum number of queries to process')
    parser.add_argument('--output-format', choices=['summary', 'detailed', 'csv'], 
                       default='detailed', help='Output format')
    parser.add_argument('--generate-sample', type=int, 
                       help='Generate a sample SQL file with N queries instead of processing')
    
    args = parser.parse_args()
    
    if args.generate_sample:
        sample_file = "sample_queries.sql"
        generate_sample_sql_file(sample_file, args.generate_sample)
        print("Sample file generated: {}".format(sample_file))
        args.filename = sample_file
    
    if not args.filename:
        parser.print_help()
        print("\nExamples:")
        print("  python query_classifier.py queries.sql")
        print("  python query_classifier.py queries.sql --use-ast")
        print("  python query_classifier.py queries.sql --max-queries 1000 --output-format summary")
        print("  python query_classifier.py --generate-sample 100")
        return
    
    print("Processing queries from: {}".format(args.filename))
    print("Using {}".format('AST parsing' if args.use_ast else 'text-based classification'))
    if args.max_queries:
        print("Maximum queries to process: {}".format(args.max_queries))
    
    templates = process_query_stream_from_file(
        args.filename, 
        use_ast=args.use_ast,
        max_queries=args.max_queries
    )
    
    print_templates(templates, args.output_format)

if __name__ == "__main__":
    main()