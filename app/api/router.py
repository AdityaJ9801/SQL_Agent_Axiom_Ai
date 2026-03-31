from fastapi import APIRouter, HTTPException, Depends
from app.core.models import QueryRequest, ExplainRequest, AggregateRequest, QueryResult
from app.services.query_manager import process_query, explain_query
from app.core.exceptions import SQLSafetyError, SQLExecutionError, SQLGenerationError
from app.db.connectors import get_database_connector
from app.config import settings

router = APIRouter()

@router.get("/health")
async def health_check():
    from app.config import settings
    return {"status": "ok", "service": "sql_query_agent", "llm_provider": "grok" if getattr(settings, "XAI_API_KEY", "") else settings.LLM_PROVIDER}

@router.post("/query", response_model=QueryResult)
async def execute_query(req: QueryRequest):
    try:
        result = await process_query(req.task_description, req.schema_context)
        return result
    except SQLSafetyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SQLExecutionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except SQLGenerationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/query/explain")
async def explain_plan(req: ExplainRequest):
    try:
        plan = await explain_query(req.task_description, req.schema_context)
        return {"explain_plan": plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/raw")
async def raw_query(sql_req: dict):
    # Admin use only according to spec
    sql = sql_req.get("sql")
    if not sql:
        raise HTTPException(status_code=400, detail="SQL required")
    
    db = get_database_connector()
    await db.connect()
    
    try:
        from app.core.safety import validate_sql_safety
        validate_sql_safety(sql) # Still validate for safety even on raw
        cols, data, rows = await db.execute(sql)
        await db.close()
        return {"columns": cols, "data": data, "total_rows": rows}
    except Exception as e:
        await db.close()
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/aggregate", response_model=QueryResult)
async def aggregate_query(req: AggregateRequest):
    # Convert the JSON dictionary spec into a text description for the AI
    spec_parts = []
    for key, value in req.spec.items():
        spec_parts.append(f"{key}: {value}")
    
    task_description = "Write an aggregation query based on this strict specification: " + ", ".join(spec_parts)
    
    try:
        # Reuse process_query to automatically fetch schema, build SQL, and execute it
        result = await process_query(task=task_description, schema=None)
        return result
    except SQLSafetyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SQLExecutionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except SQLGenerationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/run")
async def run_task(payload: dict):
    """
    Orchestrator-compatible endpoint that receives enriched task payloads.
    
    Expected payload structure:
    {
        "query": "...",  # or "task_description": "..."
        "_context": {
            "t01": { # Context agent result (ContextObject)
                "source_id": "...",
                "source_type": "...",
                "columns": [{"name": "...", "dtype": "..."}, ...],
                "row_count": 123,
                "metadata": {
                    "source": { # Original DataSource object
                        "type": "local_file",
                        "path": "...",
                        "format": "csv"
                    }
                }
            }
        }
    }
    """
    try:
        # Extract task description (could be "query" or "task_description")
        task_description = payload.get("task_description") or payload.get("query")
        if not task_description:
            raise HTTPException(status_code=400, detail="task_description or query is required")
        
        # Extract schema_context and source info from dependency results
        schema_context = None
        source_info = None
        table_name = "data_table"
        context_data = payload.get("_context", {})
        
        # Look for context agent result in dependencies
        for dep_id, dep_result in context_data.items():
            if isinstance(dep_result, dict) and "columns" in dep_result:
                # Extract source information from metadata
                metadata = dep_result.get("metadata", {})
                source_info = metadata.get("source")

                # Build schema_context from ContextObject
                columns = dep_result.get("columns", [])
                if columns:
                    # Build CREATE TABLE statement from column profiles
                    col_defs = []
                    for col in columns:
                        col_name = col.get("name", "unknown")
                        dtype = col.get("dtype", "VARCHAR")
                        semantic = col.get("semantic_type", "")

                        # Map pandas/python dtypes to SQL types
                        # semantic_type takes priority for ambiguous object columns
                        if "int" in dtype.lower():
                            sql_type = "INTEGER"
                        elif "float" in dtype.lower() or "double" in dtype.lower():
                            sql_type = "DOUBLE"
                        elif "bool" in dtype.lower():
                            sql_type = "BOOLEAN"
                        elif "date" in dtype.lower() or "time" in dtype.lower():
                            sql_type = "DATE"
                        elif semantic in ("datetime", "date", "timestamp"):
                            # object/string column that contains date strings — keep as VARCHAR
                            # but annotate so LLM knows to cast it
                            sql_type = "VARCHAR -- contains date strings, cast with CAST(col AS DATE)"
                        else:
                            sql_type = "VARCHAR"

                        col_defs.append(f'"{col_name}" {sql_type}')

                    # Derive table name — prefer the actual filename from source path,
                    # fall back to source_id. The DuckDB connector loads files using
                    # the filename stem as the table name (e.g. "sales.csv" → "sales").
                    table_name = "data_table"
                    if source_info:
                        raw_path = source_info.get("path", "")
                        # Extract stem from URL or file path: ".../bf7dccfa_sales.csv" → "bf7dccfa_sales"
                        stem = raw_path.rstrip("/").split("/")[-1].split(".")[0]
                        if stem:
                            table_name = stem
                    
                    if table_name == "data_table":
                        # Fall back to source_id last segment
                        source_id = dep_result.get("source_id", "data_table")
                        table_name = source_id.split(":")[-1].split("/")[-1].split(".")[0] or "data_table"

                    # Sanitize to valid SQL identifier
                    import re
                    table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
                    if not table_name:
                        table_name = "data_table"
                    elif table_name[0].isdigit():
                        table_name = f"t_{table_name}"

                    schema_context = f"CREATE TABLE {table_name} ({', '.join(col_defs)});"
                    break
        
        # If we have source info, load the data into the database
        db = get_database_connector()
        await db.connect()

        if source_info and settings.DB_DIALECT.lower() == "duckdb":
            src_type = source_info.get("type", "")
            file_path = source_info.get("path", "")
            file_format = source_info.get("format", "csv")

            if file_path:
                try:
                    if file_format == "csv":
                        db.conn.execute(
                            f'CREATE OR REPLACE TABLE "{table_name}" AS '
                            f"SELECT * FROM read_csv_auto('{file_path}')"
                        )
                    elif file_format == "parquet":
                        db.conn.execute(
                            f'CREATE OR REPLACE TABLE "{table_name}" AS '
                            f"SELECT * FROM read_parquet('{file_path}')"
                        )
                    else:
                        # Unknown format — create empty schema so SQL generation still works
                        try:
                            db.conn.execute(schema_context)
                        except Exception:
                            pass
                except Exception as load_err:
                    # File unreachable — create schema-only table so LLM can still generate SQL
                    try:
                        db.conn.execute(schema_context)
                    except Exception:
                        pass
            else:
                try:
                    db.conn.execute(schema_context)
                except Exception:
                    pass
        
        # Process the query using the active DB connection
        result = await process_query(task_description, schema_context, db_conn=db)
        await db.close()
        return result
        
    except SQLSafetyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SQLExecutionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except SQLGenerationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
