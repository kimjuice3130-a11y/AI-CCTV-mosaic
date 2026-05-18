#!/bin/bash

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 AI-CCTV-Mosaic 설치 스크립트${NC}"
echo ""

# 1. Python 버전 확인
echo -e "${YELLOW}1️⃣  Python 버전 확인${NC}"
python_version=$(python3 --version 2>&1)
echo "설치된 Python: $python_version"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3가 설치되지 않았습니다${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python 준비 완료${NC}"
echo ""

# 2. 가상 환경 생성
echo -e "${YELLOW}2️⃣  가상 환경 생성${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ 가상 환경 생성 완료${NC}"
else
    echo -e "${GREEN}✅ 가상 환경 이미 존재${NC}"
fi
echo ""

# 3. 가상 환경 활성화
echo -e "${YELLOW}3️⃣  가상 환경 활성화${NC}"
source venv/bin/activate
echo -e "${GREEN}✅ 가상 환경 활성화 완료${NC}"
echo ""

# 4. 의존성 설치
echo -e "${YELLOW}4️⃣  의존성 설치 중...${NC}"
pip install --upgrade pip
pip install -r backend/requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 의존성 설치 완료${NC}"
else
    echo -e "${RED}❌ 의존성 설치 실패${NC}"
    exit 1
fi
echo ""

# 5. 디렉토리 생성
echo -e "${YELLOW}5️⃣  디렉토리 생성${NC}"
mkdir -p uploads results logs
echo -e "${GREEN}✅ 디렉토리 생성 완료${NC}"
echo ""

# 6. 완료 메시지
echo -e "${GREEN}✨ 설치 완료!${NC}"
echo ""
echo -e "${YELLOW}다음 단계:${NC}"
echo ""
echo "  # 백엔드 서버 시작"
echo "  cd backend && python app.py"
echo ""
echo "  # 또는 브라우저에서 접속"
echo "  http://localhost:8000"
echo ""
