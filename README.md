# Kanana 금융 에이전트

로컬 sLLM을 활용해 국내외 최신 금융 시장 정보를 검색, 분석하고 답변하는 RAG 기반 Agent입니다. KCIF(국제금융센터) 데이터와 실시간 웹 검색을 결합하여 금융 지표에 대한 신뢰도 높은 답변을 제공합니다.

---

## 주요 기능

1. **최신 금융 데이터 누적**: KCIF 데이터를 영업일 기준(공휴일, 휴장일 제외)으로 자동 크롤링하여 Chroma Vector DB에 증분 적재하며, DB 미보유 데이터는 Tavily 웹 검색으로 보완합니다.

2. **정밀 질의응답**: LangGraph 기반의 AI 에이전트가 특정 기간·날짜·범위를 추출하고, 질문과 관련된 복수 쿼리를 생성하여 검색 정확도를 높입니다.

3. **금융 분석 보고서 생성**: 검색된 근거 문서를 바탕으로 답변 내 수치와 맥락이 근거 문서와 일치하는지 답변 품질을 검증하는 과정을 최대 2회 반복합니다. 최종적으로 금융 전문가 어조의 [요약] / [상세 분석] / [참고 문헌] 구조의 보고서를 생성합니다.

---

## 프로젝트 구조
- config.py: 환경 설정 및 .env 로딩
- setup.py: 사전 준비 확인 및 자동 초기화
- main.py: FastAPI 엔트리포인트 및 LangGraph 에이전트 그래프 정의
- model_loader.py: kanana-1.5-2.1b-instruct-2505 모델 로드
- database.py: 데이터 크롤링 및 Vector DB 증분 적재
- logger_setting.py: 로그 설정 

---

## 에이전트 파이프라인

```
질문 입력
  → Input Router           # 금융/일반/off_topic 분류
  → Multi Query Generator  # 검색 쿼리 3개 생성, 날짜 범위 추출
  → Check Availability     # DB 내 데이터 존재 여부 확인
  → RAG Searcher           # Vector DB 검색
  → Web Searcher           # DB 미보유 시 Tavily 웹 검색
  → Context Filter         # 관련 문서 필터링
  → Context Reranker       # 관련도 순 재정렬
  → Context Evaluator      # 정보 충분성 평가
  → Answer Generator       # 최종 답변 생성
  → Hallucination Grader   # 환각 검증
  → Answer Regenerator     # 환각 발견 시 최대 2회 답변 재생성
```

---

## 사전 준비
이 프로젝트는 로컬 GPU 환경에서의 구동을 권장합니다.

### 필수 조건
- Python 3.10 이상
- NVIDIA GPU 권장 (CPU 사용 시 속도 저하)
- CUDA 11.8 이상 (GPU 사용 시)
- VRAM 8GB 이상 (4bit 양자화 기준)
- [Tavily API 키](https://tavily.com) (웹 검색)
- [HuggingFace 토큰](https://huggingface.co/settings/tokens) (모델 다운로드)

## 설치와 환경설정 및 실행

### 1. 저장소 클론

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

### 4. 환경 설정

프로젝트 경로에 .env 파일 생성 후 실행

```
TAVILY_API_KEY = tvly_xxxxx
HUGGINGFACE_TOKEN = hf_xxxxx
KANANA_MODEL_PATH =        # 로컬 모델 경로 (비워두면 HuggingFace 자동 다운로드)
APP_PORT = 8000             # 서버 포트 (기본값 8000)
```
### 5. 사전 준비 확인

환경변수, CSV 데이터, Vector DB, 모델 순으로 자동 확인 및 초기화

```bash
python setup.py
```

### 6. 서버 실행
```bash
uvicorn main:api --host 0.0.0.0 --port 8000
```

---

## API 엔드포인트

### 서버 상태 확인
```bash
curl http://127.0.0.1:8000/health
```

**정상 응답:**
```json
{"ok": true}
```

---

### 질문 및 분석 요청

**요청:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "요즘 금융 시장 분위기 어때?"}'
```

**응답:**
```json
{
  "question": "요즘 금융 시장 분위기 어때?",
  "answer": "[요약]\n...\n[상세 분석]\n...\n[참고 문헌]\n...",
  "elapsed": "XX분 XX초"
}
```

---

## 로그 및 파일 저장 위치

| 항목 | 경로 | 설명 |
|---|---|---|
| 로그 | `logs/kanana_agent_YYYYMMDD_HHMMSS.log` | 질문, 답변, 답변 소요시간 |
| .CSV 데이터 | `data/kcif_articles_accumulate.csv` | 크롤링 데이터 (출처: KCIF 국제금융센터) |
| Chroma Vector DB | `data/chroma_db_bge/` | 크롤링 데이터 DB화 |

---
