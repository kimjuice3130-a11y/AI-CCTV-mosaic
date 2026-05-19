FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 파일 복사
COPY . .

# 포트 노출
EXPOSE 8000

# 서버 실행
CMD ["python", "backend/app.py"]
