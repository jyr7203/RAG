# Kanana 금융 에이전트

로컬 sLLM을 활용해 국내외 최신 금융 시장 정보를 검색, 분석하고 답변하는 RAG 기반 Agent입니다. KCIF(국제금융센터) 데이터와 실시간 웹 검색을 결합하여 금융 지표에 대한 신뢰도 높은 답변을 제공합니다.

---

## 주요 기능

1. **최신 금융 데이터 누적**: KCIF 데이터를 공휴일·휴장일을 제외한 영업일 기준으로 자동 크롤링하여 Chroma Vector DB에 증분 적재하며, DB 미보유 데이터는 Tavily 웹 검색으로 보완합니다.

2. **정밀 질의응답**: LangGraph 기반의 AI 에이전트가 특정 기간·날짜·범위를 추출하고, 복수 쿼리를 생성하여 검색 정확도를 높입니다.

3. **금융 분석 보고서 생성**: 검색된 근거 문서를 바탕으로 답변 내 수치와 맥락이 근거 문서와 일치하는지 답변 품질을 검증하는 과정을 최대 2회 반복합니다. 최종적으로 금융 전문가 어조의 [요약] / [상세 분석] / [참고 문헌] 구조의 보고서를 생성합니다.

---

## 프로젝트 구조

```
├── README.md
├── .env                  # API 키 (깃헙 비공개)
├── .gitignore
├── config.py             # 환경 설정 (경로, 모델, 디바이스 등)
├── setup.py              # 사전 준비 확인 및 자동 초기화
├── main.py               # FastAPI 서버 및 에이전트 실행
├── model_loader.py       # Kanana 모델 로더 (GPU 세대별 자동 설정)
├── database.py           # RAG DB 크롤링 및 Vector DB 관리
├── logger_setting.py     # 로그 설정
├── requirements.txt
├── data/                 # CSV 데이터 및 Vector DB (깃헙 비공개)
│   ├── kcif_articles_accumulate.csv
│   └── chroma_db_bge/
└── logs/                 # 로그 파일 (깃헙 비공개)
    └── kanana_agent_YYYYMMDD_HHMMSS.log
```

---

## 에이전트 파이프라인

```
질문 입력
  → Input Router           # 금융/일반/off_topic 분류
  → Multi Query Generator  # 검색 쿼리 3개 생성 + 날짜 범위 추출
  → Check Availability     # DB 내 데이터 존재 여부 확인
  → RAG Searcher           # Vector DB 검색
  → Web Searcher           # DB 미보유 시 Tavily 웹 검색
  → Context Filter         # 관련 문서 필터링
  → Context Reranker       # 관련도 순 재정렬
  → Context Evaluator      # 정보 충분성 평가
  → Answer Generator       # 최종 답변 생성
  → Hallucination Grader   # 환각 검증
  → Answer Regenerator     # 환각 발견 시 재생성 (최대 2회)
```

---

## 사전 준비

### 필수 조건
- Python 3.10 이상
- NVIDIA GPU 권장 (CPU도 가능하나 속도 저하)

### 1. 레포지토리 클론
```bash
git clone https://github.com/jyr7203/RAG.git
cd RAG
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. torch 설치 (환경에 맞게 선택)

**CPU 환경:**
```bash
pip install torch==2.5.1
```

**GPU 환경 — `nvidia-smi`로 CUDA 버전 확인 후 설치:**

| CUDA 버전 | 설치 명령어 |
|---|---|
| 12.1 | `pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121` |
| 12.4 | `pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124` |
| 11.8 | `pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118` |

---

## 환경 설정

`.env` 파일을 생성하고 아래 내용을 입력하세요:

```
TAVILY_API_KEY=your_tavily_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here  # HuggingFace 모델 접근 토큰
KANANA_MODEL_PATH=        # 로컬 모델 경로 (비워두면 HuggingFace 자동 다운로드)
APP_PORT=8000             # 서버 포트 (기본값 8000)
```

---

## 실행 방법

### 1. 사전 준비 확인
```bash
python setup.py
```
환경변수, CSV 데이터, Vector DB, 모델 순서로 자동 확인 및 초기화합니다.

### 2. 서버 실행
```bash
uvicorn main:api --host 0.0.0.0 --port 8000
```

### 3. Swagger UI 접속
```
http://localhost:8000/docs
```

---

## API 엔드포인트

### `GET /health`
서버 상태 확인

**응답 예시:**
```json
{"ok": true}
```

---

### `POST /ask`
금융 질문 요청

**요청:**
```json
{
  "question": "3월, 4월 엔화 환율 분석해줘"
}
```

**응답:**
```json
{
  "question": "3월, 4월 엔화 환율 분석해줘",
  "answer": "[요약]\n...\n[상세 분석]\n...\n[참고 문헌]\n...",
  "elapsed": "1분 32초"
}
```

---

## 로그 및 파일 저장 위치

| 항목 | 경로 | 설명 |
|---|---|---|
| 로그 파일 | `logs/kanana_agent_YYYYMMDD_HHMMSS.log` | 서버 시작마다 새 파일 생성, 질문/답변/소요시간 기록 |
| CSV 데이터 | `data/kcif_articles_accumulate.csv` | KCIF 크롤링 데이터 |
| Vector DB | `data/chroma_db_bge/` | Chroma 벡터 데이터베이스 |

---

## 환경별 모델 자동 설정

`model_loader.py`가 실행 환경을 자동 감지하여 최적 설정으로 로드합니다.

| 환경 | 자동 적용 설정 | 예상 속도 |
|---|---|---|
| 로컬 CPU | float32 | 매우 느림 |
| 코랩 T4 (compute 7.5) | float16 + 4bit 양자화 | 약 2~3분 |
| 코랩 A100 (compute 8.0+) | bfloat16 + 4bit 양자화 | 약 1분 |
| 로컬 RTX 3060~3090 (compute 8.6) | bfloat16 + 4bit 양자화 | 약 2~3분 |
| 로컬 RTX 4090 (compute 8.9) | bfloat16 + 4bit 양자화 | 약 1~2분 |

---

## vast.ai GPU 대여 시

```bash
git clone https://github.com/jyr7203/RAG.git
cd RAG
pip install -r requirements.txt
pip uninstall torchvision torchcodec -y   # vast.ai 환경 충돌 패키지 제거
nvidia-smi                                # CUDA 버전 확인 후 위 표에서 맞는 torch 설치
# .env 파일 생성
echo "TAVILY_API_KEY=실제키입력" > .env
echo "HUGGINGFACE_TOKEN=실제토큰입력" >> .env
python setup.py
uvicorn main:api --host 0.0.0.0 --port 8000
```
