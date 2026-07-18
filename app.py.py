"""
India TWI Data Gateway — Secure Tokenless Resolution Proxy
─────────────────────────────────────────────────────────────
"""
import os
import time
import threading
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import httpx
import requests
import rasterio
from rasterio.vrt import WarpedVRT
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from huggingface_hub import hf_hub_url

# ── Config ──────────────────────────────────────────────────────────────────
HF_TOKEN       = os.environ.get("HF_TOKEN")
CLIENT_API_KEY = os.environ.get("CLIENT_API_KEY")

TWI_REPO       = "J2003S/india-twi"
TWI_TILES      = [f"TWI_Tile{i}_COG.tif" for i in range(1, 10)]
REPO_TYPE      = "dataset"
WEB_CRS        = "EPSG:4326"
NODATA         = -9999.0

WATERSHED_REPO = "J2003S/hydrology-data-vault"
WATERSHED_FILE = "Watershed.fgb"

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is missing on Render.")

app = FastAPI(
    title="India TWI Data Gateway",
    description="Authenticated secure signing router for private TWI raster layers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Content-Length", "Accept-Ranges"],
)

def _tile_url(filename: str) -> str:
    return hf_hub_url(repo_id=TWI_REPO, filename=filename, repo_type=REPO_TYPE)

def _auth_headers() -> str:
    return f"Authorization: Bearer {HF_TOKEN}"

GDAL_OPTS = {
    "GDAL_HTTP_HEADERS": _auth_headers(),
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_USE_HEAD": "NO",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "16000000",
    "GDAL_HTTP_MULTIRANGE": "YES",
}

@contextmanager
def open_tile_as_4326(filename: str):
    url = f"/vsicurl/{_tile_url(filename)}"
    with rasterio.Env(**GDAL_OPTS):
        with rasterio.open(url) as src:
            with WarpedVRT(src, crs=WEB_CRS, src_nodata=NODATA, nodata=NODATA) as vrt:
                yield vrt

# ── Tile Footprint Indexing ──────────────────────────────────────────────────
_index_lock: threading.Lock = threading.Lock()
_tile_bounds: Dict[str, Tuple[float, float, float, float]] = {}
_index_ready: bool = False

def _build_tile_index() -> None:
    global _index_ready
    with _index_lock:
        if _index_ready:
            return
        for fname in TWI_TILES:
            try:
                with open_tile_as_4326(fname) as vrt:
                    b = vrt.bounds
                    _tile_bounds[fname] = (b.left, b.bottom, b.right, b.top)
            except Exception as e:
                print(f"[startup] WARNING: failed to read bounds for {fname}: {e}")
        _index_ready = True

@app.on_event("startup")
def _on_startup():
    _build_tile_index()

def _ensure_index() -> None:
    if not _index_ready:
        _build_tile_index()

def check_api_key(x_api_key: Optional[str]) -> str:
    if CLIENT_API_KEY and x_api_key != CLIENT_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")
    return x_api_key or "anonymous"

def _resolve_signed_url(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    raw_url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type)
    resp = requests.head(
        raw_url,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        allow_redirects=True,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.url

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/v1/manifest")
def manifest(request: Request, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    _ensure_index()
    base = str(request.base_url).rstrip("/")
    return JSONResponse({
        "crs": "EPSG:4326",
        "nodata": NODATA,
        "tiles": [
            {
                "filename": fname,
                "bounds": {"west": l, "south": b, "east": r, "north": t},
                "url": f"{base}/tiles/{fname}",
            }
            for fname, (l, b, r, t) in _tile_bounds.items()
        ],
    })

@app.get("/api/v1/resolve/watershed")
def resolve_watershed(x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    url = _resolve_signed_url(WATERSHED_REPO, WATERSHED_FILE)
    return {"url": url}

@app.get("/api/v1/resolve/tile/{filename}")
def resolve_tile(filename: str, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    if filename not in TWI_TILES:
        raise HTTPException(status_code=404, detail="Unknown tile filename.")
    url = _resolve_signed_url(TWI_REPO, filename)
    return {"url": url}