"""Co-located single writer and read-only HTTP listener with recovery gate."""
import asyncio
from contextlib import asynccontextmanager, suppress
from contextlib import closing
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .admin import load_trust
from .contracts import canonical, digest
from .publisher import Publisher
from .registry import Registry


def environment():
    path = Path(os.environ["PROXY_REGISTRY_DB"])
    inbox = Path(os.environ["PROXY_PUBLICATION_INBOX"])
    checkpoints = Path(os.environ["PROXY_CHECKPOINTS"])
    trust = load_trust(os.environ["PROXY_TRUST_FILE"])
    # Failure of recovery validation prevents the listener starting at all.
    publisher = Publisher(Registry(path, trust), inbox, checkpoints)
    if os.environ.get("PROXY_PREPUBLICATION_BACKUP"):
        destination = Path(os.environ["PROXY_PREPUBLICATION_BACKUP"])
        if not destination.exists():
            local = path.with_name("prepublication-" + trust.artifact + ".sqlite")
            with publisher.registry.connect() as source, closing(sqlite3.connect(local)) as target:
                source.backup(target)
                if target.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise ValueError("Prepublication backup integrity failure")
            os.chmod(local, 0o600)
            with local.open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
            os.chmod(destination, 0o600)
            if hashlib.sha256(local.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
                raise ValueError("Prepublication backup copy differs")
    healthy = True
    snapshot_document = publisher.registry.read().document()
    verified_at = time.monotonic()

    def refresh_publication():
        publisher.verify_floor()
        results = publisher.scan()
        publisher.verify_floor()
        return results, publisher.registry.read().document()

    @asynccontextmanager
    async def lifespan(app):
        async def consume():
            nonlocal healthy, snapshot_document, verified_at
            while True:
                try:
                    # Stage mounts can block for seconds per file. Never run their
                    # verification on the HTTP event loop or inside a reader lock.
                    results, document = await asyncio.to_thread(refresh_publication)
                    snapshot_document = document
                    verified_at = time.monotonic()
                    healthy = True
                    if results:
                        print(canonical({"publication_results": results}).decode(), flush=True)
                except Exception as error:
                    healthy = False
                    print(canonical({"publisher_unavailable": type(error).__name__}).decode(), flush=True)
                await asyncio.sleep(30)
        task = asyncio.create_task(consume())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            publisher.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/snapshot")
    async def snapshot():
        if not healthy or time.monotonic() - verified_at >= 120:
            return JSONResponse({"error": "Registry recovery gate unavailable"}, status_code=503)
        return JSONResponse(snapshot_document, headers={"Cache-Control": "no-store"})

    @app.get("/readyz")
    async def ready():
        good = healthy and time.monotonic() - verified_at < 120
        return JSONResponse({"ready": good, "revision": digest(snapshot_document) if good else None},
                            status_code=200 if good else 503)
    return app