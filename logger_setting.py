import logging
import os
from datetime import datetime
from config import Config

# 서버 시작 시 생성되는 로그 파일 경로
_LOG_FILE: str | None = None

def get_log_file() -> str:
    """서버 시작 시 타임스탬프 기반 로그 파일 경로 반환 (싱글턴)"""
    global _LOG_FILE
    if _LOG_FILE is None:
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _LOG_FILE = os.path.join(Config.LOG_DIR, f"kanana_agent_{timestamp}.log")
    return _LOG_FILE


def get_logger(name="KananaAgent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')

        # 파일 핸들러
        file_handler = logging.FileHandler(get_log_file(), encoding='utf-8')
        file_handler.setFormatter(formatter)

        # 콘솔 핸들러
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


# 정상 작동 확인
if __name__ == "__main__":
    test_log = get_logger("TestLogger")

    print("--- 로그 테스트 시작 ---")
    test_log.info("INFO 레벨 로그 정상 작동")
    test_log.warning("WARNING 레벨 로그 정상 작동")
    test_log.error("ERROR 레벨 로그 정상 작동")
    print("--- 로그 테스트 종료 ---")

    log_path = get_log_file()
    if os.path.exists(log_path):
        print(f"로그 파일 생성 확인: {log_path}")
    else:
        print("로그 파일 생성 실패")
