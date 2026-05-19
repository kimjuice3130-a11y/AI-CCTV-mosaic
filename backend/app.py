"""
FastAPI 기반 AI-CCTV 모자이크 서버
- YOLO v8 실시간 인물 감지
- 고정확도 신체 모자이크 처리
- WebSocket 실시간 스트림
- REST API 비디오 업로드
"""

import os
import sys
import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
import base64
import io

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from yolo_mosaic import YOLOMosaicProcessor, FrameTracker

# ============================================================================
# 로깅 설정
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cctv_mosaic.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 경로 설정
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# 디렉토리 생성
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# FastAPI 앱 초기화
# ============================================================================
app = FastAPI(
    title="🛡️ AI-CCTV Mosaic Server",
    description="YOLO v8 기반 고정확도 인물 모자이크 시스템",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
static_dir = BASE_DIR / "cctv-admin-human-mosaic"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ============================================================================
# 전역 상태
# ============================================================================
class ProcessingState:
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.processor: Optional[YOLOMosaicProcessor] = None
        self.device = "cuda" if self._check_cuda() else "cpu"
        self.model_name = "yolov8m"

    @staticmethod
    def _check_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

state = ProcessingState()

# ============================================================================
# 초기화
# ============================================================================
@app.on_event("startup")
async def startup():
    """앱 시작 시 YOLO 모델 로드"""
    try:
        logger.info(f"🚀 AI-CCTV Mosaic 서버 시작 (Device: {state.device})")
        state.processor = YOLOMosaicProcessor(
            model_name=state.model_name,
            device=state.device
        )
        logger.info("✅ 모든 준비 완료!")
    except Exception as e:
        logger.error(f"❌ 시작 오류: {e}")
        raise

@app.on_event("shutdown")
async def shutdown():
    """앱 종료 시 정리"""
    logger.info("🛑 서버 종료 중...")

# ============================================================================
# 루트 엔드포인트
# ============================================================================
@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지"""
    html_path = static_dir / "cctv-admin.html"
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    return """
    <html>
        <head><title>AI-CCTV Mosaic</title></head>
        <body>
            <h1>🛡️ AI-CCTV Mosaic Server</h1>
            <p>서버가 정상 작동 중입니다!</p>
            <ul>
                <li><a href="/docs">API 문서</a></li>
                <li><a href="/api/status">상태 확인</a></li>
            </ul>
        </body>
    </html>
    """

# ============================================================================
# 상태 확인 API
# ============================================================================
@app.get("/api/status")
async def status():
    """서버 상태 확인"""
    return {
        "status": "running",
        "device": state.device,
        "model": state.model_name,
        "cuda_available": state.device == "cuda",
        "active_tasks": len(state.tasks),
        "timestamp": time.time()
    }

# ============================================================================
# WebSocket 실시간 스트림
# ============================================================================
@app.websocket("/ws/process-stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket 실시간 프레임 처리
    
    클라이언트 메시지:
    {
        "type": "frame",
        "data": "base64_encoded_jpeg",
        "config": {
            "confidence": 0.5,
            "mosaic_strength": 15,
            "blur_kernel": 21
        }
    }
    """
    await websocket.accept()
    frame_count = 0
    detections_total = 0
    start_time = time.time()

    try:
        logger.info("📡 WebSocket 연결 수립")

        while True:
            # 메시지 수신
            message = await websocket.receive_json()
            
            if message.get("type") == "frame":
                # Base64 디코딩
                try:
                    frame_data = base64.b64decode(message.get("data", ""))
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is None:
                        continue

                    # 설정 가져오기
                    config = message.get("config", {})
                    confidence = config.get("confidence", 0.5)
                    mosaic_strength = config.get("mosaic_strength", 15)
                    blur_kernel = config.get("blur_kernel", 21)
                    padding = config.get("padding", 15)

                    # 모자이크 처리
                    process_start = time.time()
                    frame_output, detections = state.processor.process_frame(
                        frame,
                        confidence=confidence,
                        mosaic_strength=mosaic_strength,
                        blur_kernel=blur_kernel,
                        padding=padding
                    )
                    process_time = (time.time() - process_start) * 1000

                    # 결과 인코딩
                    ret, buffer = cv2.imencode('.jpg', frame_output, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')

                    frame_count += 1
                    detections_total += len(detections)

                    # FPS 계산
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0

                    # 응답 송신
                    await websocket.send_json({
                        "type": "result",
                        "frame_id": frame_count,
                        "data": frame_b64,
                        "detections": len(detections),
                        "fps": round(fps, 2),
                        "process_time_ms": round(process_time, 2),
                        "persons": [
                            {
                                "id": i,
                                "confidence": round(d['confidence'], 3),
                                "bbox": d['bbox']
                            }
                            for i, d in enumerate(detections)
                        ]
                    })

                except Exception as e:
                    logger.error(f"프레임 처리 오류: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })

            elif message.get("type") == "close":
                logger.info(f"📊 처리 완료 - 총 {frame_count}프레임, {detections_total}명 감지")
                break

    except Exception as e:
        logger.error(f"WebSocket 오류: {e}")
    finally:
        await websocket.close()
        logger.info("📡 WebSocket 연결 종료")

# ============================================================================
# 비디오 업로드 및 배치 처리
# ============================================================================
@app.post("/api/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.1, le=0.9),
    mosaic_strength: int = Query(15, ge=5, le=30),
    blur_kernel: int = Query(21, ge=5, le=51),
    background_tasks: BackgroundTasks = None
):
    """
    비디오 파일 업로드 및 처리
    
    Args:
        file: 업로드 비디오 파일
        confidence: YOLO 감지 신뢰도
        mosaic_strength: 모자이크 강도
        blur_kernel: 블러 커널 크기
    """
    try:
        # 파일 ID 생성
        task_id = str(uuid.uuid4())
        timestamp = time.time()
        task_id_with_time = f"{timestamp}_{task_id}"

        # 파일 저장
        file_ext = Path(file.filename).suffix
        input_path = UPLOAD_DIR / f"{task_id_with_time}_input{file_ext}"
        output_path = OUTPUT_DIR / f"{task_id_with_time}_output.mp4"

        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"📥 파일 업로드: {file.filename} ({len(content) / 1024 / 1024:.2f}MB)")

        # 작업 정보 저장
        state.tasks[task_id_with_time] = {
            "status": "processing",
            "input": str(input_path),
            "output": str(output_path),
            "confidence": confidence,
            "mosaic_strength": mosaic_strength,
            "blur_kernel": blur_kernel,
            "progress": 0,
            "start_time": timestamp,
            "end_time": None
        }

        # 백그라운드 작업 등록
        if background_tasks:
            background_tasks.add_task(
                process_video_task,
                task_id_with_time,
                str(input_path),
                str(output_path),
                confidence,
                mosaic_strength,
                blur_kernel
            )

        return {
            "task_id": task_id_with_time,
            "status": "processing",
            "message": "영상 처리 중입니다",
            "file_size_mb": len(content) / 1024 / 1024
        }

    except Exception as e:
        logger.error(f"업로드 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# ============================================================================
# 비디오 처리 백그라운드 작업
# ============================================================================
def process_video_task(task_id: str, input_path: str, output_path: str,
                      confidence: float, mosaic_strength: int, blur_kernel: int):
    """비디오 처리 백그라운드 작업"""
    try:
        logger.info(f"🎬 비디오 처리 시작: {task_id}")

        def progress_callback(progress: float, detections: int):
            if task_id in state.tasks:
                state.tasks[task_id]["progress"] = progress
                state.tasks[task_id]["detections"] = detections

        # 처리 수행
        result = state.processor.process_video(
            input_path,
            output_path,
            confidence=confidence,
            mosaic_strength=mosaic_strength,
            blur_kernel=blur_kernel,
            progress_callback=progress_callback
        )

        # 작업 완료
        state.tasks[task_id]["status"] = "completed"
        state.tasks[task_id]["end_time"] = time.time()
        state.tasks[task_id]["result"] = result

        logger.info(f"✅ 처리 완료: {task_id} - {result['total_persons']}명 감지")

    except Exception as e:
        logger.error(f"❌ 처리 오류: {e}")
        if task_id in state.tasks:
            state.tasks[task_id]["status"] = "error"
            state.tasks[task_id]["error"] = str(e)

# ============================================================================
# 작업 상태 조회
# ============================================================================
@app.get("/api/task-status/{task_id}")
async def task_status(task_id: str):
    """작업 상태 조회"""
    if task_id not in state.tasks:
        return JSONResponse(
            status_code=404,
            content={"error": "작업을 찾을 수 없습니다"}
        )

    task = state.tasks[task_id]
    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", 0),
        "detections": task.get("detections", 0)
    }

    if task["status"] == "completed":
        response["download_url"] = f"/api/download/{task_id}"
        response["result"] = task.get("result")

    if task["status"] == "error":
        response["error"] = task.get("error")

    return response

# ============================================================================
# 결과 다운로드
# ============================================================================
@app.get("/api/download/{task_id}")
async def download(task_id: str):
    """처리된 비디오 다운로드"""
    if task_id not in state.tasks:
        return JSONResponse(
            status_code=404,
            content={"error": "작업을 찾을 수 없습니다"}
        )

    task = state.tasks[task_id]
    output_path = task.get("output")

    if not output_path or not Path(output_path).exists():
        return JSONResponse(
            status_code=404,
            content={"error": "출력 파일을 찾을 수 없습니다"}
        )

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"mosaic_{task_id}.mp4"
    )

# ============================================================================
# 작업 목록 조회
# ============================================================================
@app.get("/api/tasks")
async def list_tasks(limit: int = Query(10, ge=1, le=100)):
    """최근 작업 목록 조회"""
    tasks = list(state.tasks.items())
    tasks.sort(key=lambda x: x[1].get("start_time", 0), reverse=True)

    return {
        "total": len(state.tasks),
        "tasks": [
            {
                "task_id": task_id,
                "status": task.get("status"),
                "progress": task.get("progress", 0),
                "detections": task.get("detections", 0),
                "start_time": task.get("start_time")
            }
            for task_id, task in tasks[:limit]
        ]
    }

# ============================================================================
# 작업 취소
# ============================================================================
@app.post("/api/cancel/{task_id}")
async def cancel_task(task_id: str):
    """작업 취소"""
    if task_id not in state.tasks:
        return JSONResponse(
            status_code=404,
            content={"error": "작업을 찾을 수 없습니다"}
        )

    task = state.tasks[task_id]
    if task["status"] == "processing":
        task["status"] = "cancelled"
        return {"message": "작업이 취소되었습니다"}
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "처리 중이 아닌 작업은 취소할 수 없습니다"}
        )

# ============================================================================
# 메인
# ============================================================================
if __name__ == "__main__":
    logger.info(f"""
    ╔══════════════════════════════════════════════╗
    ║     🛡️  AI-CCTV Mosaic Server v1.0.0       ║
    ║  YOLO v8 기반 고정확도 인물 모자이크        ║
    ║══════════════════════════════════════════════║
    ║  Device: {state.device.upper():40}║
    ║  Model:  {state.model_name:40}║
    ║  API:    http://0.0.0.0:8000/docs           ║
    ║  WebSocket: ws://0.0.0.0:8000/ws/...        ║
    ╚══════════════════════════════════════════════╝
    """)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
