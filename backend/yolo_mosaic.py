"""
YOLO v8 기반 모자이크 처리 엔진
- 인물 감지 (Ultralytics YOLO v8)
- 신체 경계 정확 감지
- 정교한 모자이크 적용
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Callable, Optional
import logging

try:
    from ultralytics import YOLO
except ImportError:
    print("⚠️ ultralytics 미설치. 설치 필요: pip install ultralytics")

logger = logging.getLogger(__name__)


class YOLOMosaicProcessor:
    """YOLO v8 기반 모자이크 처리"""

    def __init__(self, model_name: str = "m", device: str = "auto"):
        """
        초기화
        
        Args:
            model_name: 모델 크기 ("n", "s", "m", "l", "x")
            device: 디바이스 ("cpu", "cuda", "auto")
        """
        self.model_name = model_name
        self.device = device if device != "auto" else self._detect_device()
        
        try:
            self.model = YOLO(f"yolov8{model_name}.pt")
            self.model.to(self.device)
            logger.info(f"✅ YOLO v8{model_name} 모델 로드 완료 ({self.device})")
        except Exception as e:
            logger.error(f"❌ YOLO 모델 로드 실패: {e}")
            self.model = None

    def _detect_device(self) -> str:
        """GPU 사용 가능 여부 확인"""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except:
            return "cpu"

    def detect_persons(self, frame: np.ndarray, confidence: float = 0.5) -> list:
        """
        프레임에서 인물 감지
        
        Args:
            frame: 입력 프레임
            confidence: 신뢰도 임계값 (0~1)
        
        Returns:
            감지된 인물 박스 리스트 [(x1, y1, x2, y2, conf), ...]
        """
        if self.model is None:
            return []

        try:
            results = self.model(frame, conf=confidence, verbose=False)
            detections = []

            for result in results:
                for box in result.boxes:
                    # COCO 데이터셋에서 클래스 0 = person
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        detections.append((x1, y1, x2, y2, conf))

            return detections

        except Exception as e:
            logger.error(f"❌ 감지 실패: {e}")
            return []

    def apply_mosaic(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mosaic_strength: int = 15,
        blur_kernel: int = 21,
        extend_padding: int = 10,
    ) -> np.ndarray:
        """
        박스 영역에 모자이크 적용
        
        Args:
            frame: 입력 프레임
            bbox: 박스 좌표 (x1, y1, x2, y2)
            mosaic_strength: 모자이크 블록 크기 (5~30)
            blur_kernel: 블러 커널 크기 (5~51, 홀수)
            extend_padding: 박스 주변 여백
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # 여백 확장
        x1 = max(0, x1 - extend_padding)
        y1 = max(0, y1 - extend_padding)
        x2 = min(w, x2 + extend_padding)
        y2 = min(h, y2 + extend_padding)

        # 모자이크 적용
        roi = frame[y1:y2, x1:x2]

        # 방법 1: 블록 모자이크
        h_roi, w_roi = roi.shape[:2]
        small = cv2.resize(roi, (w_roi // mosaic_strength, h_roi // mosaic_strength))
        mosaic_roi = cv2.resize(small, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)

        # 방법 2: 가우시안 블러로 부드럽게
        blurred_roi = cv2.GaussianBlur(mosaic_roi, (blur_kernel, blur_kernel), 0)

        # 페더링 (엣지 부드럽게)
        mask = np.ones((h_roi, w_roi), dtype=np.float32)
        cv2.circle(
            mask,
            (w_roi // 2, h_roi // 2),
            min(w_roi, h_roi) // 2,
            1,
            -1
        )

        # 경계 페더링
        for i in range(min(10, min(w_roi, h_roi) // 10)):
            cv2.circle(
                mask,
                (w_roi // 2, h_roi // 2),
                min(w_roi, h_roi) // 2 - i,
                (1 - (i / 10)) ** 2,
                1
            )

        # 블렌드
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = np.stack([mask] * 3, axis=2)
        result_roi = (blurred_roi * mask + roi * (1 - mask)).astype(np.uint8)

        frame[y1:y2, x1:x2] = result_roi
        return frame

    def process_frame(
        self,
        frame: np.ndarray,
        confidence: float = 0.5,
        mosaic_strength: int = 15,
        blur_kernel: int = 21,
        extend_padding: int = 10,
    ) -> Tuple[np.ndarray, Dict]:
        """
        단일 프레임 처리
        
        Returns:
            (처리된 프레임, 통계 딕셔너리)
        """
        # 감지
        detections = self.detect_persons(frame, confidence)

        # 모자이크 적용
        output = frame.copy()
        for x1, y1, x2, y2, conf in detections:
            output = self.apply_mosaic(
                output,
                (x1, y1, x2, y2),
                mosaic_strength,
                blur_kernel,
                extend_padding,
            )

        return output, {"detections": len(detections)}

    def process_image(
        self,
        input_path: str,
        output_path: str,
        confidence: float = 0.5,
        mosaic_strength: int = 15,
        blur_kernel: int = 21,
        extend_padding: int = 10,
    ) -> Dict:
        """이미지 파일 처리"""
        try:
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError("이미지 로드 실패")

            output, stats = self.process_frame(
                img, confidence, mosaic_strength, blur_kernel, extend_padding
            )
            cv2.imwrite(output_path, output)

            logger.info(f"✅ 이미지 처리 완료: {output_path}")
            return {"detections": stats["detections"]}

        except Exception as e:
            logger.error(f"❌ 이미지 처리 실패: {e}")
            raise

    def process_video(
        self,
        input_path: str,
        output_path: str,
        confidence: float = 0.5,
        mosaic_strength: int = 15,
        blur_kernel: int = 21,
        extend_padding: int = 10,
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """
        비디오 파일 처리
        
        Args:
            input_path: 입력 비디오 경로
            output_path: 출력 비디오 경로
            progress_callback: 진행률 콜백 함수
        """
        try:
            cap = cv2.VideoCapture(input_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            frame_count = 0
            total_detections = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                output, stats = self.process_frame(
                    frame, confidence, mosaic_strength, blur_kernel, extend_padding
                )
                out.write(output)

                frame_count += 1
                total_detections += stats["detections"]

                if progress_callback:
                    progress_callback(frame_count, total_frames, total_detections)

            cap.release()
            out.release()

            logger.info(f"✅ 비디오 처리 완료: {output_path}")
            return {
                "total_frames": total_frames,
                "total_detections": total_detections,
                "fps": fps,
            }

        except Exception as e:
            logger.error(f"❌ 비디오 처리 실패: {e}")
            raise


class FrameProcessor:
    """프레임 처리 보조 클래스"""

    def __init__(self, processor: YOLOMosaicProcessor):
        self.processor = processor

    def process_frame(self, frame: np.ndarray, config: Dict) -> Tuple[np.ndarray, Dict]:
        """설정을 사용한 프레임 처리"""
        return self.processor.process_frame(
            frame,
            confidence=config.get("confidence", 0.5),
            mosaic_strength=config.get("mosaic_strength", 15),
            blur_kernel=config.get("blur_kernel", 21),
            extend_padding=config.get("extend_padding", 10),
        )
