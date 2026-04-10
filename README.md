# Kanana 금융 에이전트

Kanana 모델(kakaocorp/kanana-1.5-2.1b-instruct-2505) 기반 국제 금융 정보 RAG 에이전트입니다.

## 폴더 구조

```
├── README.md
├── .env                  # API 키
├── .gitignore
├── config.py             # 환경 설정
├── setup.py              # 사전 준비 확인
├── main.py               # 에이전트 실행
├── model_loader.py       # Kanana 모델 로더
├── database.py           # RAG DB 크롤링 및 관리
├── logger_setting.py     # 로그 설정
├── requirements.txt
├── data/                 # CSV 데이터 및 Vector DB
└── logs/                 # 로그 파일
```

## 시작하기

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

GPU 사용 시 torch CUDA 버전으로 별도 설치:
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

### 2. 환경변수 설정

`.env` 파일을 생성하고 API 키를 입력하세요:

```
TAVILY_API_KEY=your_tavily_api_key_here
KANANA_MODEL_PATH=  # 로컬 모델 경로 (비워두면 HuggingFace 자동 다운로드)
```

### 3. 사전 준비 확인

```bash
python setup.py
```

### 4. 에이전트 실행

```bash
python main.py
```

## 환경별 설정

| 환경 | 설정 |
|---|---|


