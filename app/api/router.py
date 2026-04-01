import os
import re
import asyncio
import tempfile
from fastapi import APIRouter, HTTPException
from app.core.models import QueryRequest, ExplainRequest, AggregateRequest, QueryResult
from app.services.query_manager import process_query, explain_query
from app.core.exceptions import SQLSafetyError, SQLExecutionError, SQLGenerationError
from app.db.connectors import get_database_connector
from app.config import settings

router = APIRouter()

_HEX_PREFIX = re.compile(r'^[0-9a-f]{6,}_', re.IGNORECASE)
_AZURE_BLOB_RE = re.compile(r'https://[^.]+\.blob\.core\.windows\.net/([^/?]+)/(.+?)(\?.*)?$')


async def _download_azure_blob(blob_url: str, suffix: str = ".csv") -> str | None:
    """Download a private Azure Blob to a temp file using the Python SDK.

    DuckDB's bundled libcurl fails SSL verification on python:3.11-slim because
    ca-certificates may be missing. Python's azure-storage-blob uses the certifi
    CA bundle and works reliably in all container environments.

    Returns the temp file path (caller must delete it), or None on failure.
    """
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        return None

    m = _AZURE_BLOB_RE.match(blob_url)
    if not m:
        return None

    container, blob_name = m.group(1), m.group(2)

    def _download() -> bytes:
        from azure.storage.blob import BlobServiceClient
        client = BlobServiceClient.from_connection_string(conn_str)
        return client.get_blob_client(container=container, blob=blob_name).download_blob().readall()

    try:
        content = await asyncio.get_event_loop().run_in_executor(None, _download)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(content)
        tmp.close()
        return tmp.name
    except Exception as exc:
        print(f"Warning: Azure SDK blob download failed for {blob_url}: {exc}")
        return None


def _clean_table_name(raw: str) -> str:
    """Derive a clean SQL table name from a raw filename stem or source_id segment.

    Examples:
        "95b1734e_sales"  → "sales"
        "bf7dccfa_sales"  → "sales"
        "1346c6ad6297338b" → "dataset"   (pure hash, no meaningful suffix)
        "sales"           → "sales"
    """
    # Strip leading hex hash prefix: "95b1734e_sales" → "sales"
    clean = _HEX_PREFIX.sub("", raw)
    # If nothing meaningful remains (pure hash), fall back
    if not clean or clean == raw and re.fullmatch(r'[0-9a-f]+', raw, re.IGNORECASE):
        clean = "dataset"
    # Sanitize: only alphanumeric + underscore
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', clean).strip("_") or "dataset"
    # No leading digit
    if clean[0].isdigit():
        clean = f"t_{clean}"
    return clean

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
    """
    try:
        # Extract task description (could be "query" or "task_description")
        task_description = payload.get("task_description") or payload.get("query")
        if not task_description:
            raise HTTPException(status_code=400, detail="task_description or query is required")
        
        # Extract schema_context and source info from dependency results
        schema_parts = []
        sources_to_load = []
        context_data = payload.get("_context", {})
        
        # Look for context agent results in dependencies
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
                        if "int" in dtype.lower():
                            sql_type = "INTEGER"
                        elif "float" in dtype.lower() or "double" in dtype.lower() or "decimal" in dtype.lower():
                            sql_type = "DOUBLE"
                        elif "bool" in dtype.lower():
                            sql_type = "BOOLEAN"
                        elif "date" in dtype.lower() or "time" in dtype.lower():
                            sql_type = "DATE"
                        elif semantic in ("datetime", "date", "timestamp"):
                            sql_type = "VARCHAR -- contains date strings, cast with CAST(col AS DATE)"
                        else:
                            sql_type = "VARCHAR"

                        col_defs.append(f'"{col_name}" {sql_type}')

                    # Derive a clean, predictable table name
                    table_name = "dataset"
                    if source_info:
                        raw_path = source_info.get("path", "")
                        stem = raw_path.rstrip("/").split("/")[-1].split(".")[0]
                        if stem:
                            table_name = _clean_table_name(stem)

                    if table_name == "dataset":
                        source_id = dep_result.get("source_id", "")
                        stem = source_id.split(":")[-1].split("/")[-1].split(".")[0]
                        if stem:
                            table_name = _clean_table_name(stem)

                    schema_parts.append(f"CREATE TABLE IF NOT EXISTS \"{table_name}\" ({', '.join(col_defs)});")
                    if source_info:
                        sources_to_load.append((table_name, source_info))
        
        schema_context = "\n".join(schema_parts) if schema_parts else None

        # If we have source info, load the data into the database
        db = get_database_connector()
        await db.connect()

        if sources_to_load and settings.DB_DIALECT.lower() == "duckdb":
            for table_name, source_info in sources_to_load:
                file_path = source_info.get("path", "")
                file_format = source_info.get("format", "csv")

                if not file_path:
                    continue

                loaded = False
                tmp_path = None
                try:
                    # For Azure Blob URLs, download to a local temp file first.
                    # This uses Python's azure-storage-blob SDK (and its certifi CA bundle)
                    # instead of DuckDB's bundled libcurl, which fails SSL verification on
                    # python:3.11-slim containers with "Problem with the SSL CA cert".
                    is_azure_blob = bool(_AZURE_BLOB_RE.match(file_path))
                    if is_azure_blob:
                        suffix = f".{file_format}" if file_format else ".csv"
                        tmp_path = await _download_azure_blob(file_path, suffix=suffix)
                        effective_path = tmp_path if tmp_path else file_path
                        if tmp_path:
                            print(f"Info: Downloaded Azure blob to temp file: {file_path} → {tmp_path}")
                        else:
                            print(f"Warning: Azure SDK download failed, falling back to direct URL: {file_path}")
                    else:
                        effective_path = file_path

                    if file_format == "parquet":
                        await db.execute(
                            f"CREATE OR REPLACE TABLE \"{table_name}\" AS "
                            f"SELECT * FROM read_parquet('{effective_path}')"
                        )
                    else:
                        # Default to CSV for unknown formats
                        await db.execute(
                            f"CREATE OR REPLACE TABLE \"{table_name}\" AS "
                            f"SELECT * FROM read_csv_auto('{effective_path}')"
                        )
                    loaded = True
                except Exception as load_err:
                    print(f"Warning: Failed to load data for {table_name}: {load_err}")
                finally:
                    # Always clean up temp files
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                # If data load failed, create an empty schema table so the LLM
                # still knows the column structure (it will just return no rows).
                if not loaded:
                    matching_stmts = [s for s in schema_parts if f'"{table_name}"' in s]
                    for stmt in matching_stmts:
                        try:
                            await db.execute(stmt)
                        except Exception as e:
                            print(f"Warning: Failed fallback CREATE TABLE: {e}")

        # Ensure all tables are created even if data loading was skipped or failed
        for stmt in schema_parts:
            try:
                await db.execute(stmt)
            except Exception:
                pass
        
        # Process the query using the active DB connection
        try:
            result = await process_query(task_description, schema_context, db_conn=db)
            await db.close()
            return result
        except Exception as e:
            await db.close()
            raise e
        
    except SQLSafetyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SQLExecutionError as e:
        print(f"SQLExecutionError in /run: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except SQLGenerationError as e:
        print(f"SQLGenerationError in /run: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Unexpected error in /run: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
