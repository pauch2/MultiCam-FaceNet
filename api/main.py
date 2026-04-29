import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles

from api.auth import authenticate_user, create_access_token
from api.routers import stream, faces, client_cam, logs, admin, cameras, users
from api.camera_pool import CameraPool
from database.session import init_db, migrate_db, seed_users, seed_cameras


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing database…")
    init_db()
    migrate_db()
    seed_users()
    seed_cameras()

    app.state.pool = CameraPool()   # multi-camera pool (replaces single camera)
    app.state.camera = None         # kept for legacy stream.py compat (not used)

    try:
        yield
    finally:
        app.state.pool.stop_all()
        try:
            from api.batch_embedder import stop_batch_embedder
            stop_batch_embedder()
        except Exception:
            pass


app = FastAPI(title="Face ID API", version="1.0", lifespan=lifespan)

app.include_router(stream.router)
app.include_router(faces.router)
app.include_router(client_cam.router)
app.include_router(logs.router)
app.include_router(admin.router)
app.include_router(cameras.router)
app.include_router(users.router)

_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── Authentication ────────────────────────────────────────────────────────────

@app.post("/token", tags=["Authentication"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": user["username"], "role": user["role"], "display_name": user.get("display_name", user["username"])})
    return {"access_token": token, "token_type": "bearer"}


# ── File serving ──────────────────────────────────────────────────────────────

@app.get("/detections/img", include_in_schema=False)
def serve_detection_img(path: str):
    import pathlib, config
    from fastapi import HTTPException

    storage_root = (pathlib.Path(config.BASE_DIR) / "storage" / "detections").resolve()

    # Paths are stored as absolute by frame_processor; handle both absolute + relative.
    # On Windows paths may have backslashes; pathlib handles this natively.
    path = path.replace("\\", "/")   # normalize any double-escaped backslashes from JSON
    candidate = pathlib.Path(path)
    if not candidate.is_absolute():
        candidate = (pathlib.Path(config.BASE_DIR) / candidate)
    candidate = candidate.resolve()

    # Security check — must be inside storage/detections/
    try:
        candidate.relative_to(storage_root)
    except ValueError:
        # Log what we got to help debug path mismatches
        import logging
        logging.getLogger("main").warning(
            "serve_detection_img blocked: %s  (root: %s)", candidate, storage_root)
        raise HTTPException(403, "Access denied")

    if not candidate.is_file():
        raise HTTPException(404, f"Image not found: {candidate.name}")

    return FileResponse(str(candidate), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return "<h1>Face ID</h1><p><a href='/ui'>/ui</a> · <a href='/docs'>/docs</a></p>"


@app.get("/ui", include_in_schema=False)
def ui_page():
    return FileResponse(os.path.join(_static_dir, "ui.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)