from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.router import router
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-install DuckDB extensions if using DuckDB
    if settings.DB_DIALECT.lower() == "duckdb":
        import duckdb
        try:
            conn = duckdb.connect(database=":memory:")
            conn.execute("INSTALL httpfs; LOAD httpfs;")
            # Also install azure and aws extensions if possible, as they are often needed for remote files
            try:
                conn.execute("INSTALL azure; LOAD azure;")
            except: pass
            try:
                conn.execute("INSTALL aws; LOAD aws;")
            except: pass
            conn.close()
        except Exception as e:
            print(f"Warning: Failed to pre-install DuckDB extensions: {e}")
    yield

app = FastAPI(
    title="SQL Query Agent",
    description="Translates NL tasks into SQL, ensures safety, and executes against target databases",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
