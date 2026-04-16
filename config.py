# 환경 설정
import torch
import os
from dotenv import load_dotenv

# .env 파일
load_dotenv()

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 데이터 경로
    DATA_DIR  = os.path.join(BASE_DIR, "data")
    DATA_PATH = os.path.join(DATA_DIR, "kcif_articles_accumulate.csv")
    DB_PATH   = os.path.join(DATA_DIR, "chroma_db_bge")

    # 로그 경로
    LOG_DIR  = os.path.join(BASE_DIR, "logs")
    LOG_FILE = os.path.join(LOG_DIR, "kanana_agent.log")

    # API 키 (.env 로드)
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY가 .env 파일에 설정되지 않았습니다.")
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

    # 모델 Kanana 설정 - 로컬 모델 파일 경로
    KANANA_MODEL_PATH = os.getenv("KANANA_MODEL_PATH", None)
    LLM_MODEL = KANANA_MODEL_PATH if KANANA_MODEL_PATH else "kakaocorp/kanana-1.5-2.1b-instruct-2505"

    # 디바이스 설정
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # 임베딩
    EMBED_MODEL_NAME   = "BAAI/bge-m3"
    EMBED_MODEL_KWARGS = {'device': DEVICE}
    EMBED_ENCODE_KWARGS = {'normalize_embeddings': True}

    # 서버 포트
    APP_PORT = int(os.getenv("APP_PORT", 8000))

    @classmethod
    def ensure_dirs(cls):
        """필요한 디렉토리 자동 생성"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)

# 디렉토리 자동 생성
Config.ensure_dirs()

# 정상 작동 확인
if __name__ == "__main__":
    print(f"BASE_DIR  : {Config.BASE_DIR}")
    print(f"DATA_DIR  : {Config.DATA_DIR}")
    print(f"LOG_DIR   : {Config.LOG_DIR}")
    print(f"LLM_MODEL : {Config.LLM_MODEL}")
    print(f"DEVICE    : {Config.DEVICE}")
    print(f"TAVILY_API_KEY 로드: {'Yes' if Config.TAVILY_API_KEY else 'No'}")
