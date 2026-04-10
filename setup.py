# ============================================================
# setup.py  ―  사전 준비 확인 및 초기화
# ============================================================
import os
import sys
from config import Config
from logger_setting import get_logger

log = get_logger("Setup")


def check_env() -> bool:
    """필수 환경변수 확인"""
    log.info("[ 1/4 ] 환경변수 확인 중...")
    if not Config.TAVILY_API_KEY:
        log.error("❌ TAVILY_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return False
    log.info("✅ 환경변수 정상 확인")
    return True


def check_model() -> bool:
    """Kanana 모델 로드 가능 여부 확인"""
    log.info("[ 2/4 ] Kanana 모델 확인 중...")
    try:
        from model_loader import KananaModel
        KananaModel.get_model()
        log.info("✅ Kanana 모델 로드 완료")
        return True
    except Exception as e:
        log.error(f"❌ 모델 로드 실패: {e}")
        return False


def check_csv() -> bool:
    """RAG용 CSV 데이터 파일 확인"""
    log.info("[ 3/4 ] CSV 데이터 파일 확인 중...")
    if os.path.exists(Config.DATA_PATH):
        import pandas as pd
        df = pd.read_csv(Config.DATA_PATH)
        log.info(f"✅ CSV 파일 확인 완료 (총 {len(df)}개 행, 경로: {Config.DATA_PATH})")
        return True
    else:
        log.warning(f"⚠️  CSV 파일 없음: {Config.DATA_PATH}")
        log.warning("→ sync_csv_data()를 실행하여 데이터를 수집합니다.")
        try:
            from database import sync_csv_data
            sync_csv_data()
            log.info("✅ CSV 데이터 수집 완료")
            return True
        except Exception as e:
            log.error(f"❌ CSV 데이터 수집 실패: {e}")
            return False


def check_vector_db() -> bool:
    """Vector DB 확인 및 없을 시 자동 생성"""
    log.info("[ 4/4 ] Vector DB 확인 중...")
    if os.path.exists(Config.DB_PATH) and os.listdir(Config.DB_PATH):
        log.info(f"✅ Vector DB 확인 완료 (경로: {Config.DB_PATH})")
        return True
    else:
        log.warning(f"⚠️  Vector DB 없음: {Config.DB_PATH}")
        log.warning("→ update_vector_db()를 실행하여 DB를 생성합니다.")
        try:
            from database import update_vector_db
            update_vector_db()
            log.info("✅ Vector DB 생성 완료")
            return True
        except Exception as e:
            log.error(f"❌ Vector DB 생성 실패: {e}")
            return False


def run_setup() -> bool:
    """전체 사전 준비 실행"""
    log.info("=" * 50)
    log.info("Kanana Agent 사전 준비 시작")
    log.info("=" * 50)

    checks = [
        check_env,
        check_csv,
        check_vector_db,
        check_model,
    ]

    for check in checks:
        if not check():
            log.error("❌ 사전 준비 실패. 위 오류를 확인하세요.")
            return False

    log.info("=" * 50)
    log.info("✅ 모든 사전 준비 완료! main.py를 실행하세요.")
    log.info("=" * 50)
    return True


if __name__ == "__main__":
    success = run_setup()
    sys.exit(0 if success else 1)
