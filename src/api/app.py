# src/api/app.py
import torch
import threading
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.api.usecases import REGISTRY, ROOT

app = FastAPI(title="Vision Monitor Demo")

DEFAULT_VIDEOS = {
    "retail": "data/shop_763.mp4",
    "luggage": "data/avss2007_luggage.mp4",
    "traffic": "data/traffic.mp4",
}


class AnalyzeRequest(BaseModel):
    usecase: str
    video_path: str


class Runner:
    def __init__(self, usecase_key, video_path):
        if usecase_key not in REGISTRY:
            raise ValueError(f"Use case inconnu : {usecase_key} "
                             f"(dispo : {', '.join(REGISTRY)})")
        path = video_path
        if not Path(path).is_absolute():
            path = ROOT / path
        self.uc = REGISTRY[usecase_key](path)
        self.video_path = str(path)
        self.usecase_key = usecase_key

        self._jpeg = None
        self._lock = threading.Lock()
        self.monitor = {}
        self.stats = {}
        self.running = False
        self.finished = False

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.video_path)
        frame_id = 0
        target_dt = 1.0 / 25.0
        read_fails = 0

        while self.running:
            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                read_fails += 1
                if read_fails > 30:
                    print(f"[run] STOP frame_id={frame_id} "
                          f"pos={cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} "
                          f"total={cap.get(cv2.CAP_PROP_FRAME_COUNT):.0f}")
                    break
                continue
            read_fails = 0

            annotated, monitor, stats = self.uc.process(frame, frame_id)
            self.monitor = monitor
            self.stats = stats

            disp = cv2.resize(annotated, (1100, 619))
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                with self._lock:
                    self._jpeg = buf.tobytes()

            frame_id += 1

            dt = time.perf_counter() - t0
            if dt < target_dt:
                time.sleep(target_dt - dt)

        cap.release()
        self.running = False
        self.finished = True

    def latest(self):
        with self._lock:
            return self._jpeg


_runner: Runner | None = None
_thread: threading.Thread | None = None


@app.get("/usecases")
def usecases():
    return {k: {"name": cls.display_name,
                "video": DEFAULT_VIDEOS.get(k, "")}
            for k, cls in REGISTRY.items()}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    global _runner, _thread
    if _runner and _runner.running:
        _runner.running = False
        if _thread:
            _thread.join(timeout=2.0)
    try:
        _runner = Runner(req.usecase, req.video_path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Erreur d'initialisation : {e}")

    _thread = threading.Thread(target=_runner.run, daemon=True)
    _thread.start()
    return {"status": "started", "usecase": req.usecase,
            "video": _runner.video_path, "fps": _runner.uc.fps}


def _mjpeg():
    last = None
    while _runner and (_runner.running or not _runner.finished):
        jpg = _runner.latest()
        if jpg is not None and jpg is not last:
            last = jpg
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpg + b"\r\n")
        else:
            time.sleep(0.005)


@app.get("/stream")
def stream():
    if _runner is None:
        raise HTTPException(400, "Aucune analyse. POST /analyze d'abord.")
    return StreamingResponse(
        _mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/monitor")
def monitor():
    if _runner is None:
        return JSONResponse({"status": "idle"})
    return JSONResponse({"running": _runner.running,
                         "finished": _runner.finished,
                         "usecase": _runner.usecase_key,
                         "monitor": _runner.monitor})


@app.get("/stats")
def stats():
    if _runner is None:
        return JSONResponse({"status": "idle"})
    return JSONResponse({"running": _runner.running,
                         "finished": _runner.finished,
                         "usecase": _runner.usecase_key, **_runner.stats})


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "src" / "api" / "demo.html").read_text(encoding="utf-8")
