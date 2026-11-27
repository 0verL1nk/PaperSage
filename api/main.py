import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent import __version__
from agent.adapters.orm import run_migrations
from agent.application.research_workspace import research_workspace_service
from agent.logging_utils import configure_application_logging

from .context_memory_routes import context_memory_router
from .eval_routes import eval_router
from .feedback_routes import evidence_click_router, feedback_findings_router
from .routes import router
from .session_command_routes import session_command_router
from .suggestion_routes import suggestion_router


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    run_migrations()
    yield
    research_workspace_service.close()


app = FastAPI(title="PaperSage API", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(context_memory_router, prefix="/api/v1")
app.include_router(suggestion_router, prefix="/api/v1")
app.include_router(eval_router, prefix="/api/v1")
app.include_router(session_command_router, prefix="/api/v1")
app.include_router(feedback_findings_router, prefix="/api/v1")
app.include_router(evidence_click_router, prefix="/api/v1")

RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
WEB_DIST = RESOURCE_ROOT / "web" / "dist"
ROOT_STATIC_FILES = {
    "favicon.svg": WEB_DIST / "favicon.svg",
    "icons.svg": WEB_DIST / "icons.svg",
    "papersage-mark.svg": WEB_DIST / "papersage-mark.svg",
}
if (WEB_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def spa_fallback(path: str) -> FileResponse:
    # The SPA entry must never come from a heuristic browser cache: after an
    # in-place desktop update a stale entry references chunk hashes the new
    # backend no longer ships, breaking lazy imports until the cache clears.
    candidate = ROOT_STATIC_FILES.get(path)
    if candidate and candidate.is_file():
        return FileResponse(candidate, headers={"Cache-Control": "no-store"})
    if path == "index.html":
        candidate = WEB_DIST / "index.html"
        if candidate.is_file():
            return FileResponse(candidate, headers={"Cache-Control": "no-store"})
    return FileResponse(WEB_DIST / "index.html", headers={"Cache-Control": "no-store"})


def run() -> None:
    configure_application_logging(
        debug_mode=os.getenv("PAPERSAGE_DESKTOP") == "1",
        logger_name="api",
    )
    uvicorn.run("api.main:app", host="127.0.0.1", port=int(os.getenv("PAPERSAGE_PORT", "8000")), reload=False)


if __name__ == "__main__":
    run()
