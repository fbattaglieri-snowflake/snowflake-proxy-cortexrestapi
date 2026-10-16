"""Read-only registry listener; initialize database through local admin first."""
import os
from pathlib import Path
import sqlite3
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from .contracts import parse_snapshot, strict_json


def read_snapshot(path):
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2)
    try:
        raw = connection.execute("SELECT body FROM state WHERE id=1").fetchone()[0]
        if len(raw.encode()) > 4 * 1024 * 1024:
            raise ValueError("Snapshot too large")
        return parse_snapshot(strict_json(raw))
    finally:
        connection.close()


def create_app():
    path = Path(os.environ["PROXY_REGISTRY_DB"]).resolve(strict=True)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/snapshot")
    def snapshot():
        return JSONResponse(read_snapshot(path).document(), headers={"Cache-Control": "no-store"})

    @app.get("/readyz")
    def ready():
        try:
            snapshot()
            return {"ready": True}
        except (ValueError, OSError, sqlite3.Error, TypeError):
            return JSONResponse({"ready": False}, status_code=503)

    return app