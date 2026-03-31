import time
import sqlglot
from typing import Dict, Any, Tuple, Optional
from app.config import settings
from app.core.exceptions import SQLSafetyError, SQLExecutionError
from app.core.safety import validate_sql_safety
from app.db.connectors import get_database_connector
from app.llm.generator import generate_sql, correct_sql
from app.llm.summarizer import generate_summary
from app.core.models import QueryResult

def transpile_sql(sql: str, to_dialect: str) -> str:
    """Transpile SQL to target dialect using SQLGlot, with DATEADD normalization."""
    valid_dialects = ["postgresql", "mysql", "sqlite", "duckdb", "snowflake", "bigquery"]
    if to_dialect.lower() not in valid_dialects:
        return sql

    try:
        # Try tsql first (catches DATEADD, ISNULL, etc.), fall back to generic
        for read_dialect in ("tsql", "mysql", None):
            try:
                result = sqlglot.transpile(sql, read=read_dialect, write=to_dialect.lower())[0]
                if result:
                    return result
            except Exception:
                continue
        return sql
    except Exception:
        return sql

def calculate_summary_stats(data: list) -> dict:
    """Calculate basic min/max/mean for numeric columns."""
    stats = {}
    if not data:
        return stats
        
    sample = data[0]
    numeric_cols = [k for k, v in sample.items() if isinstance(v, (int, float))]
    
    for col in numeric_cols:
        values = [row[col] for row in data if row.get(col) is not None]
        if values:
            stats[col] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values)
            }
            
    return stats

async def process_query(task: str, schema: Optional[str], db_conn=None) -> QueryResult:
    start_time = time.time()
    dialect = settings.DB_DIALECT
    
    db = db_conn
    close_db_after = False
    if db is None:
        db = get_database_connector()
        await db.connect()
        close_db_after = True
    
    if not schema:
        schema = await db.get_schema_context()
    
    # If still no schema, use a default demo schema as fallback
    if not schema or schema.strip() == "":
        schema = """
        CREATE TABLE sales (product_id INT, product_name VARCHAR, category VARCHAR, quantity INT, price DECIMAL, revenue DECIMAL, sale_date DATE, region VARCHAR);
        CREATE TABLE orders (order_id INT, customer_id VARCHAR, order_date DATE, region VARCHAR, status VARCHAR, total_amount DECIMAL);
        CREATE TABLE products (product_id INT, product_name VARCHAR, category VARCHAR, unit_price DECIMAL, stock_quantity INT, supplier_id VARCHAR);
        CREATE TABLE customers (customer_id VARCHAR, customer_name VARCHAR, email VARCHAR, region VARCHAR, signup_date DATE, total_orders INT);
        CREATE TABLE order_items (order_item_id INT, order_id INT, product_id INT, quantity INT, unit_price DECIMAL, total_price DECIMAL);
        """
        
        # Initialize the demo schema in the database to prevent catalog errors
        for statement in schema.strip().split(';'):
            if statement.strip():
                try:
                    await db.execute(statement.strip())
                except Exception:
                    pass
                    
        # Add some dummy data so queries return meaningful results in demo mode
        dummy_data = """
        INSERT INTO sales VALUES (1, 'Laptop', 'Electronics', 5, 1200.00, 6000.00, '2023-10-01', 'North America');
        INSERT INTO sales VALUES (2, 'Smartphone', 'Electronics', 12, 800.00, 9600.00, '2023-10-02', 'Europe');
        INSERT INTO sales VALUES (3, 'Desk Chair', 'Furniture', 8, 150.00, 1200.00, '2023-10-03', 'North America');
        INSERT INTO orders VALUES (1001, 'CUST-01', '2023-10-01', 'North America', 'Shipped', 6000.00);
        INSERT INTO orders VALUES (1002, 'CUST-02', '2023-10-02', 'Europe', 'Processing', 9600.00);
        INSERT INTO orders VALUES (1003, 'CUST-03', '2023-10-03', 'North America', 'Delivered', 1200.00);
        """
        for statement in dummy_data.strip().split(';'):
            if statement.strip():
                try:
                    await db.execute(statement.strip())
                except Exception:
                    pass
    else:
        # Schema was provided — ensure tables exist in this connection.
        # Use IF NOT EXISTS so we never overwrite a table already loaded with real data.
        for statement in schema.strip().split(';'):
            stmt = statement.strip()
            if stmt.upper().startswith("CREATE TABLE") and "IF NOT EXISTS" not in stmt.upper():
                stmt = stmt.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1)
            if stmt:
                try:
                    await db.execute(stmt)
                except Exception:
                    pass
        
    raw_sql = await generate_sql(task, schema, dialect)
    
    async def try_execute(query_sql: str) -> Tuple[str, list, list, int]:
        """Attempts to run SQL. Returns (final_sql, columns, data, total_rows). Raises exception on failure"""
        validate_sql_safety(query_sql)
        native_sql = transpile_sql(query_sql, dialect)
        cols, result_data, rows_total = await db.execute(native_sql, settings.MAX_RESULT_ROWS)
        return native_sql, cols, result_data, rows_total

    try:
        final_sql, columns, data, total_rows = await try_execute(raw_sql)
    except Exception as e:
        # Retry once
        corrected_sql = await correct_sql(raw_sql, str(e), task, schema, dialect)
        try:
            final_sql, columns, data, total_rows = await try_execute(corrected_sql)
        except Exception as e2:
            if close_db_after:
                await db.close()
            raise SQLExecutionError(f"Failed after retry. First Error: {e}, Retry Error: {e2}")

    # Generate summary & compress
    summary_stats = calculate_summary_stats(data)
    nl_summary = await generate_summary(task, data)
    
    has_more = total_rows > settings.MAX_RESULT_ROWS if total_rows is not None else len(data) == settings.MAX_RESULT_ROWS
    
    exec_time_ms = int((time.time() - start_time) * 1000)
    
    if close_db_after:
        await db.close()
    
    return QueryResult(
        sql_generated=final_sql,
        dialect=dialect,
        rows_returned=len(data),
        rows_total=total_rows,
        execution_time_ms=exec_time_ms,
        columns=columns,
        data_preview=data,
        summary_stats=summary_stats,
        natural_language_summary=nl_summary,
        has_more=has_more,
        result_token_estimate=len(str(data)) // 4  # rough estimate
    )

async def explain_query(task: str, schema: Optional[str]) -> str:
    dialect = settings.DB_DIALECT
    db = get_database_connector()
    await db.connect()
    
    if not schema:
        schema = await db.get_schema_context()
    
    # Validate that we have schema information
    if not schema or schema.strip() == "":
        await db.close()
        raise SQLExecutionError(
            "No schema context available. Please provide schema_context in the request, "
            "or ensure the database has tables loaded. For DuckDB, place CSV/Parquet files "
            f"in '{settings.DUCKDB_DATA_PATH}' directory."
        )
        
    raw_sql = await generate_sql(task, schema, dialect)
    validate_sql_safety(raw_sql)
    native_sql = transpile_sql(raw_sql, dialect)
    
    plan = await db.explain(native_sql)
    await db.close()
    
    return plan
