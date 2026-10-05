"""
FastAPI live stream.   Run:  uvicorn stream_server:app --port 8000
  GET /health
  WS  /ws?delay=0.5      -> one JSON message per 600-hour window (predict -> detect -> diagnose -> adapt)
"""
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from data_generator import generate, TRAIN_END, WINDOW
from drift_engine import DualRunner, make_windows

app = FastAPI(title="Live drift stream")
_state = {}

def _load():
    if "df" not in _state:
        _state["df"] = generate()
    return _state["df"]

@app.get("/health")
def health():
    return {"status": "ok", "rows": len(_load()), "window": WINDOW}

@app.websocket("/ws")
async def stream(ws: WebSocket, delay: float = Query(0.5, ge=0.0, le=10.0)):
    await ws.accept()
    df = _load()
    wins = make_windows(len(df), TRAIN_END, WINDOW)
    try:
        runner = await asyncio.to_thread(DualRunner, df, TRAIN_END, WINDOW)       # pre-trains the source model
        await ws.send_json({"type": "ready", "n_windows": len(wins), "reference_mae": runner.reference_mae, "ks_thr": runner.ks_thr})
        for s, e in wins:
            out = await asyncio.to_thread(runner.step, s, e)
            out["type"] = "window"
            out["truth_regime"] = df.regime.iloc[s]          # ONLY for shading the demo chart; the detector never sees it
            await ws.send_json(out)
            await asyncio.sleep(delay)
        await ws.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
