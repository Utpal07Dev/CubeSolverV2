"""
Carbon-Fiber Rubik's Cube Solver -- FastAPI backend.
"""

from __future__ import annotations

import base64
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

try:
    import kociemba
except Exception:  # pragma: no cover
    kociemba = None

app = FastAPI(title="Carbon Fiber Rubik's Cube Solver", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FACE_ORDER = ["U", "R", "F", "D", "L", "B"]
WORK_WIDTH = 640

# Color mapping for annotations (BGR format for OpenCV)
COLOR_BGR = {
    "white": (220, 220, 220),
    "yellow": (0, 230, 255),
    "green": (0, 255, 128),
    "blue": (255, 100, 0),
    "red": (30, 30, 240),
    "orange": (0, 140, 255),
}


# --------------------------------------------------------------------------- #
#  Colour classification
# --------------------------------------------------------------------------- #
def classify_pixel(h: float, s: float, v: float, b: float, g: float, r: float) -> str:
    """Classify a single pixel. OpenCV hue is 0..179."""
    if v < 55:
        return "white"
    if s < 55 and v < 110:
        return "white"
    if s < 45 and v >= 110:
        return "white"

    if h <= 8 or h >= 172:
        return "red"
    if 9 <= h <= 19:
        return "orange"
    if 20 <= h <= 48:
        if (g - r) > 22 or h >= 42:
            return "green"
        return "yellow"
    if 49 <= h <= 92:
        return "green"
    if 93 <= h <= 140:
        return "blue"
    return "red"


def rim_color(bgr: np.ndarray, box: Tuple[int, int, int, int]) -> str:
    """Sample the plastic rim (outer annulus) of one facelet and vote."""
    x, y, w, h = box
    patch = bgr[y : y + h, x : x + w]
    if patch.size == 0:
        return "white"

    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    ph, pw = patch.shape[:2]

    yy, xx = np.mgrid[0:ph, 0:pw]
    cy, cx = (ph - 1) / 2.0, (pw - 1) / 2.0
    nx = np.abs(xx - cx) / max(cx, 1e-6)
    ny = np.abs(yy - cy) / max(cy, 1e-6)
    cheb = np.maximum(nx, ny)
    mask = (cheb >= 0.55) & (cheb <= 0.93)
    if not mask.any():
        mask = np.ones((ph, pw), dtype=bool)

    votes: Dict[str, float] = {}
    hs = hsv[..., 0][mask].astype(np.float32)
    ss = hsv[..., 1][mask].astype(np.float32)
    vs = hsv[..., 2][mask].astype(np.float32)
    bs = patch[..., 0][mask].astype(np.float32)
    gs = patch[..., 1][mask].astype(np.float32)
    rs = patch[..., 2][mask].astype(np.float32)

    for i in range(hs.shape[0]):
        name = classify_pixel(hs[i], ss[i], vs[i], bs[i], gs[i], rs[i])
        weight = 0.35 if vs[i] < 55 else 1.0
        votes[name] = votes.get(name, 0.0) + weight

    if not votes:
        return "white"
    return max(votes.items(), key=lambda kv: kv[1])[0]


# --------------------------------------------------------------------------- #
#  Contour-based facelet detection
# --------------------------------------------------------------------------- #
def _candidate_boxes(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    edges = cv2.Canny(gray, 30, 90)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 6
    )
    combined = cv2.bitwise_or(edges, thr)
    combined = cv2.dilate(combined, np.ones((3, 3), np.uint8), iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_area = float(bgr.shape[0] * bgr.shape[1])
    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.004 or area > img_area * 0.20:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        if len(approx) < 4 or len(approx) > 6:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if w == 0 or h == 0:
            continue
        aspect = w / float(h)
        if aspect < 0.65 or aspect > 1.55:
            continue
        if area / float(w * h) < 0.55:
            continue
        boxes.append((x, y, w, h))
    return boxes


def _dedupe(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: List[Tuple[int, int, int, int]] = []
    for b in boxes:
        bx, by, bw, bh = b
        bcx, bcy = bx + bw / 2, by + bh / 2
        dup = False
        for k in kept:
            kx, ky, kw, kh = k
            kcx, kcy = kx + kw / 2, ky + kh / 2
            if abs(bcx - kcx) < max(bw, kw) * 0.5 and abs(bcy - kcy) < max(bh, kh) * 0.5:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


def _pick_nine(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    if len(boxes) <= 9:
        return boxes
    areas = np.array([b[2] * b[3] for b in boxes], dtype=np.float32)
    median = float(np.median(areas))
    ranked = sorted(boxes, key=lambda b: abs(b[2] * b[3] - median))
    return ranked[:9]


def _grid_fallback(bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]):
    h, w = bgr.shape[:2]
    if boxes:
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
    else:
        m = int(min(w, h) * 0.08)
        x0, y0, x1, y1 = m, m, w - m, h - m
    cw, ch = (x1 - x0) / 3.0, (y1 - y0) / 3.0
    out = []
    for r in range(3):
        for c in range(3):
            out.append(
                (
                    int(x0 + c * cw + cw * 0.06),
                    int(y0 + r * ch + ch * 0.06),
                    int(cw * 0.88),
                    int(ch * 0.88),
                )
            )
    return out


def sort_3x3(boxes: List[Tuple[int, int, int, int]]):
    by_y = sorted(boxes, key=lambda b: b[1] + b[3] / 2)
    rows = [by_y[0:3], by_y[3:6], by_y[6:9]]
    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b[0] + b[2] / 2))
    return ordered


def normalize_to_pitch(bgr: np.ndarray, ordered: List[Tuple[int, int, int, int]]):
    h_img, w_img = bgr.shape[:2]
    cxs = [b[0] + b[2] / 2 for b in ordered]
    cys = [b[1] + b[3] / 2 for b in ordered]
    dx = [abs(cxs[r * 3 + c + 1] - cxs[r * 3 + c]) for r in range(3) for c in range(2)]
    dy = [abs(cys[(r + 1) * 3 + c] - cys[r * 3 + c]) for r in range(2) for c in range(3)]
    pitch_x = float(np.median(dx)) if dx else 0.0
    pitch_y = float(np.median(dy)) if dy else 0.0

    out = []
    for i, (x, y, w, h) in enumerate(ordered):
        want_w = max(w, int(round(pitch_x * 0.96)))
        want_h = max(h, int(round(pitch_y * 0.96)))
        nx = int(max(0, min(cxs[i] - want_w / 2, w_img - 2)))
        ny = int(max(0, min(cys[i] - want_h / 2, h_img - 2)))
        nw = max(2, min(want_w, w_img - nx))
        nh = max(2, min(want_h, h_img - ny))
        out.append((nx, ny, nw, nh))
    return out


def annotate_detection(bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]], colors: List[str]) -> str:
    """Produce a base64 JPEG overlaying sampled boxes and classified colors."""
    annotated = bgr.copy()
    for i, (box, color) in enumerate(zip(boxes, colors)):
        x, y, w, h = box
        bgr_col = COLOR_BGR.get(color, (0, 255, 0))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), bgr_col, 2)
        cv2.putText(
            annotated,
            f"{i+1}:{color[0].upper()}",
            (x + 4, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"{i+1}:{color[0].upper()}",
            (x + 4, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


def read_face(data: bytes, include_debug_img: bool = False) -> Dict:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    if img.shape[1] > WORK_WIDTH:
        scale = WORK_WIDTH / float(img.shape[1])
        img = cv2.resize(img, (WORK_WIDTH, int(img.shape[0] * scale)))

    boxes = _dedupe(_candidate_boxes(img))
    boxes = _pick_nine(boxes)
    method = "contours"
    if len(boxes) != 9:
        boxes = _grid_fallback(img, boxes)
        method = "grid-fallback"

    ordered = normalize_to_pitch(img, sort_3x3(boxes))
    colors = [rim_color(img, b) for b in ordered]

    res = {"colors": colors, "method": method, "boxes": ordered}
    if include_debug_img:
        res["annotated_image"] = annotate_detection(img, ordered, colors)
    return res


# --------------------------------------------------------------------------- #
#  Facelet string + solving
# --------------------------------------------------------------------------- #
def build_facelets(faces: Dict[str, List[str]]) -> str:
    centers = {face: faces[face][4] for face in FACE_ORDER}
    if len(set(centers.values())) != 6:
        raise HTTPException(
            status_code=422,
            detail=f"Centers are not 6 unique colours: {centers}. Fix center facelets manually.",
        )
    color_to_face = {color: face for face, color in centers.items()}
    out = []
    for face in FACE_ORDER:
        for color in faces[face]:
            if color not in color_to_face:
                raise HTTPException(status_code=422, detail=f"Unknown colour: {color}")
            out.append(color_to_face[color])
    s = "".join(out)
    for face in FACE_ORDER:
        if s.count(face) != 9:
            raise HTTPException(
                status_code=422,
                detail=f"Face {face} appears {s.count(face)} times (must be 9). Adjust colors.",
            )
    return s


@app.get("/api/health")
def health():
    return {"status": "ok", "solver": kociemba is not None}


@app.post("/api/scan-face")
async def scan_face(file: UploadFile = File(...)):
    """Single-face detection endpoint for interactive live scanning."""
    content = await file.read()
    return read_face(content, include_debug_img=True)


@app.post("/api/solve")
async def solve(
    up: UploadFile = File(...),
    right: UploadFile = File(...),
    front: UploadFile = File(...),
    down: UploadFile = File(...),
    left: UploadFile = File(...),
    back: UploadFile = File(...),
):
    uploads = {
        "U": up,
        "R": right,
        "F": front,
        "D": down,
        "L": left,
        "B": back,
    }
    faces: Dict[str, List[str]] = {}
    debug: Dict[str, str] = {}
    for key, upload in uploads.items():
        result = read_face(await upload.read())
        faces[key] = result["colors"]
        debug[key] = result["method"]

    facelets = build_facelets(faces)

    if kociemba is None:
        return JSONResponse(
            {
                "faces": faces,
                "facelets": facelets,
                "detection": debug,
                "solution": None,
                "error": "kociemba not installed",
            },
            status_code=200,
        )
    try:
        solution = kociemba.solve(facelets)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Unsolvable state: {exc}")

    moves = solution.split()
    return {
        "faces": faces,
        "facelets": facelets,
        "detection": debug,
        "solution": solution,
        "moves": moves,
        "moveCount": len(moves),
    }


@app.post("/api/solve-facelets")
async def solve_facelets(payload: Dict[str, str]):
    facelets = (payload.get("facelets") or "").strip().upper()
    if len(facelets) != 54:
        raise HTTPException(status_code=400, detail="facelets must be 54 characters")
    if kociemba is None:
        raise HTTPException(status_code=500, detail="kociemba library not installed on server")
    try:
        solution = kociemba.solve(facelets)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Unsolvable state: {exc}")
    moves = solution.split()
    return {
        "facelets": facelets,
        "solution": solution,
        "moves": moves,
        "moveCount": len(moves),
    }


@app.get("/")
def index():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "Place index.html next to main.py"}