from app.config import settings
from app.core.exceptions import SQLGenerationError
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

_DIALECT_HINTS: dict[str, str] = {
    "duckdb": (
        "DuckDB-specific rules:\n"
        "- Date arithmetic: use `date_add(col, INTERVAL N DAY/MONTH/YEAR)` or `col + INTERVAL '7' DAY` — NEVER use DATEADD()\n"
        "- Date truncation: `date_trunc('month', col)` — NOT TRUNC()\n"
        "- Current date: `CURRENT_DATE` or `today()`\n"
        "- String → date cast: `CAST(col AS DATE)` or `col::DATE` — NEVER compare VARCHAR date columns directly to date literals without casting\n"
        "- Extract parts: `EXTRACT(YEAR FROM col)` or `YEAR(col)`\n"
        "- String functions: `regexp_matches()`, `string_split()`, `list_aggregate()`\n"
        "- If a column dtype is VARCHAR but contains dates (e.g. '2024-01-05'), always cast: `CAST(col AS DATE)`"
    ),
    "postgresql": (
        "PostgreSQL-specific rules:\n"
        "- Date arithmetic: `col + INTERVAL '7 days'` — NEVER use DATEADD()\n"
        "- Date truncation: `date_trunc('month', col)`\n"
        "- Current date: `CURRENT_DATE`\n"
        "- Cast: `col::DATE` or `CAST(col AS DATE)`"
    ),
    "sqlite": (
        "SQLite-specific rules:\n"
        "- Date arithmetic: `date(col, '+7 days')` — NEVER use DATEADD()\n"
        "- Current date: `date('now')`\n"
        "- No native DATE type — dates are stored as TEXT, use `date()` functions"
    ),
}


def generate_prompt(task: str, schema: str, dialect: str, max_rows: int = 1000) -> str:
    dialect_hint = _DIALECT_HINTS.get(dialect.lower(), f"Use standard {dialect} SQL syntax.")
    return f"""You are a senior SQL analyst. Generate a single SQL SELECT query to answer the user's request.

Database dialect: {dialect}
Available tables and schema:
{schema}

User request: {task}

{dialect_hint}

General rules:
1. Generate ONLY a SELECT query — no INSERT, UPDATE, DELETE, DROP, CREATE
2. Always include LIMIT {max_rows} unless user explicitly asks for all rows
3. Use column names exactly as shown in schema
4. For aggregations, always include meaningful column aliases
5. Prefer CTEs over nested subqueries for readability
6. Output ONLY the SQL query, no explanation, no markdown fences"""

async def _call_llm(prompt: str) -> str:
    provider = settings.LLM_PROVIDER.lower()

    # Auto-select
    if getattr(settings, "XAI_API_KEY", None) and provider != "ollama":
        provider = "grok"
    elif getattr(settings, "GROQ_API_KEY", None) and provider != "ollama":
        provider = "groq"

    messages = [
        SystemMessage(content="You are an SQL generator. Only reply with the raw SQL code."),
        HumanMessage(content=prompt)
    ]
    
    try:
        if provider == "groq":
            llm = ChatGroq(api_key=settings.GROQ_API_KEY, model_name=settings.GROQ_MODEL, temperature=0.0)
        elif provider == "grok" or getattr(settings, "XAI_API_KEY", None):
            llm = ChatOpenAI(api_key=settings.XAI_API_KEY, base_url="https://api.x.ai/v1", model="grok-2-latest", temperature=0.0)
        elif provider == "ollama":
            llm = ChatOllama(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL, temperature=0.0)
        elif provider == "mcp":
            # MCP (Model Context Protocol)
            llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model_name="gpt-4", temperature=0.0)
        elif provider == "openai":
            if not getattr(settings, "OPENAI_API_KEY", None):
                raise SQLGenerationError("OPENAI_API_KEY is missing for OpenAI provider")
            llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model_name=getattr(settings, "OPENAI_MODEL", "gpt-4o"), temperature=0.0)
        else:
            raise SQLGenerationError(f"Unsupported LLM Provider: {provider}")

        response = await llm.ainvoke(messages)
        return response.content.strip().strip("```sql").strip("`").strip("```").strip()
    except Exception as e:
        raise SQLGenerationError(f"LLM call failed: {e}")

async def generate_sql(task: str, schema: str, dialect: str) -> str:
    prompt = generate_prompt(task, schema, dialect, settings.MAX_RESULT_ROWS)
    sql = await _call_llm(prompt)
    if not sql:
        raise SQLGenerationError("LLM returned empty SQL")
    return sql

async def correct_sql(sql: str, error_msg: str, task: str, schema: str, dialect: str) -> str:
    dialect_hint = _DIALECT_HINTS.get(dialect.lower(), f"Use standard {dialect} SQL syntax.")
    prompt = f"""The following SQL query for '{task}' failed on {dialect}.

Original query:
{sql}

Error:
{error_msg}

Schema:
{schema}

{dialect_hint}

Fix the query. Common causes:
- DATEADD() is not valid in {dialect} — use the dialect-specific date function above
- VARCHAR date columns must be cast before comparison: CAST(col AS DATE)
- Function names are case-sensitive in some dialects

Output ONLY the corrected raw SQL. No markdown, no explanation.
"""
    corrected_sql = await _call_llm(prompt)
    if not corrected_sql:
        raise SQLGenerationError("LLM returned empty SQL on correction attempt")
    return corrected_sql
