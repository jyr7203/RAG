import logging
import os
from config import Config

def get_logger(name="KananaAgent"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')

        # 파일 핸들러 (logs/ 디렉토리)
        file_handler = logging.FileHandler(Config.LOG_FILE, encoding='utf-8')
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

    if os.path.exists(Config.LOG_FILE):
        print(f"로그 파일 생성 확인: {Config.LOG_FILE}")
    else:
        print("로그 파일 생성 실패")
