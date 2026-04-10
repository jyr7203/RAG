# 필요한 라이브러리
import os
import re
import time
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  
from langchain_core.documents import Document
from config import Config
from logger_setting import get_logger

log = get_logger("Database")


# 신규 데이터 크롤링 및 업데이트
def sync_csv_data():
    log.info("KCIF 신규 데이터 확인 및 크롤링 시작...")

    page_url = "https://www.kcif.or.kr/search?rpt_cd=002001001&odr=new&pg={}"

    # 기존 데이터 로드 및 기준일 설정
    if os.path.exists(Config.DATA_PATH):
        df_existing = pd.read_csv(Config.DATA_PATH)
        df_existing['date'] = pd.to_datetime(df_existing['date'])
        last_date   = df_existing['date'].max()
        cutoff_date = last_date + pd.Timedelta(days=1)
        log.info(f"기존 최신 날짜: {last_date.strftime('%Y-%m-%d')}. 이후 데이터부터 수집합니다.")
    else:
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        df_existing = pd.DataFrame(columns=["date", "content"])
        today       = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = today.replace(year=today.year - 3)
        log.info(f"기존 데이터 없음. 3개년치({cutoff_date.strftime('%Y-%m-%d')}) 수집을 시작합니다.")

    new_dates, new_contents = [], []
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })

    stop_crawling = False
    for page in tqdm(range(1, 201), desc="페이지 스캔 중"):
        if stop_crawling:
            break
        try:
            res = session.get(page_url.format(page), timeout=15)
            res.encoding = 'utf-8'
            soup  = BeautifulSoup(res.text, "html.parser")
            items = soup.select("li.list_item")
            if not items:
                break

            for item in items:
                date_tag    = item.select_one("div.list_top span:nth-of-type(2)")
                content_tag = item.select_one("a em.gray_txt")
                if not date_tag or not content_tag:
                    continue

                date_obj = pd.to_datetime(date_tag.get_text(strip=True))
                if date_obj < cutoff_date:
                    log.info(f"{date_obj.strftime('%Y-%m-%d')} 도달: 신규 데이터 수집 완료.")
                    stop_crawling = True
                    break

                text = " ".join(content_tag.get_text(separator=" ", strip=True).split()).strip()
                new_dates.append(date_obj)
                new_contents.append(text)
                time.sleep(0.3)

        except Exception as e:
            log.error(f"페이지 {page} 오류: {e}")
            break

    if new_dates:
        df_new   = pd.DataFrame({"date": new_dates, "content": new_contents})
        df_final = pd.concat([df_new, df_existing], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=['date', 'content'], keep='first')
        df_final = df_final.sort_values(by='date', ascending=False)
        df_final.to_csv(Config.DATA_PATH, index=False, encoding="utf-8-sig")
        log.info(f"업데이트 완료! 신규: {len(df_new)}개 / 총: {len(df_final)}개")
    else:
        log.info("✅ 최신 상태입니다. 추가할 데이터가 없습니다.")


# 전처리 및 섹션 분할
def split_sections(text):
    sections = {}
    if not isinstance(text, str) or not text:
        return sections

    news_match = re.search(r"■\s*주요\s*뉴스\s*:\s*(.*?)(?=■\s*국제|$)", text, re.S)
    if news_match:
        sections["주요뉴스"] = news_match.group(1).strip()

    market_match = re.search(r"■\s*국제금융시장\s*:\s*(.*)", text, re.S)
    if market_match:
        sections["국제금융시장"] = market_match.group(1).strip()

    return sections


def custom_financial_preprocessor(df_input):
    final_chunks = []
    for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc="문서 변환 중"):
        if pd.isna(row['content']):
            continue

        doc_date = pd.to_datetime(row['date'])
        date_str = doc_date.strftime('%Y-%m-%d')
        sections = split_sections(row["content"])

        base_metadata = {"date": date_str, "year": doc_date.year, "source": "KCIF"}

        for section_name, text in sections.items():
            if not text:
                continue
            meta = base_metadata.copy()
            meta["section"] = section_name

            if section_name == "주요뉴스":
                cleaned_text = text.replace(" ○ ", "\n- ").strip()
                content      = f"[{date_str} 주요뉴스 요약]\n{cleaned_text}"
                meta["item"] = "종합뉴스"
                final_chunks.append(Document(page_content=content, metadata=meta))
            elif section_name == "국제금융시장":
                cleaned_text = text.replace(" ○ ", "\n[지표] ").replace(" ※ ", "\n[참고] ").strip()
                content      = f"[{date_str} 국제금융시장 지표 및 분석]\n{cleaned_text}"
                meta["item"] = "금융지표_종합"
                final_chunks.append(Document(page_content=content, metadata=meta))

    return final_chunks


# Vector DB 업데이트
def update_vector_db():
    log.info("Vector DB 증분 업데이트 시작...")

    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBED_MODEL_NAME,
        model_kwargs=Config.EMBED_MODEL_KWARGS,
        encode_kwargs=Config.EMBED_ENCODE_KWARGS,
    )

    # DB 로드 혹은 생성
    if os.path.exists(Config.DB_PATH):
        vector_db    = Chroma(persist_directory=Config.DB_PATH, embedding_function=embeddings)
        all_metadata = vector_db.get(include=['metadatas'])['metadatas']
        if all_metadata:
            last_db_date = pd.to_datetime(max([m['date'] for m in all_metadata]))
            log.info(f"DB 내 마지막 데이터 날짜: {last_db_date.strftime('%Y-%m-%d')}")
        else:
            last_db_date = None
    else:
        log.info("기존 DB가 없습니다. 새로 생성합니다.")
        os.makedirs(Config.DB_PATH, exist_ok=True)
        vector_db    = Chroma(persist_directory=Config.DB_PATH, embedding_function=embeddings)
        last_db_date = None

    # CSV 로드 및 신규 데이터 필터링
    if not os.path.exists(Config.DATA_PATH):
        log.error("CSV 파일이 없습니다. 크롤링을 먼저 진행하세요.")
        return None

    df_full        = pd.read_csv(Config.DATA_PATH)
    df_full['date'] = pd.to_datetime(df_full['date'])
    df_new = df_full[df_full['date'] > last_db_date].copy() if last_db_date is not None else df_full

    if df_new.empty:
        log.info("✅ DB가 이미 최신 상태입니다.")
        return vector_db

    # 전처리 및 적재
    new_docs = custom_financial_preprocessor(df_new)
    if new_docs:
        batch_size = 50
        for i in tqdm(range(0, len(new_docs), batch_size), desc="Chroma 적재 중"):
            batch = new_docs[i: i + batch_size]
            vector_db.add_documents(batch)
        log.info(f"업데이트 완료! (신규 {len(new_docs)}개 청크 추가)")

    return vector_db


if __name__ == "__main__":
    sync_csv_data()
    vector_db = update_vector_db()

    if vector_db:
        results = vector_db.get(include=['documents', 'metadatas'])
        if results['metadatas']:
            sorted_indices = sorted(
                range(len(results['metadatas'])),
                key=lambda i: results['metadatas'][i]['date'],
                reverse=True,
            )
            latest_idx  = sorted_indices[0]
            latest_meta = results['metadatas'][latest_idx]
            latest_doc  = results['documents'][latest_idx]

            print("\n" + "=" * 50)
            print(" [DB LATEST DATA SAMPLE]")
            print(f" 날짜: {latest_meta.get('date')}")
            print(f" 섹션: {latest_meta.get('section')}")
            print(f"  항목: {latest_meta.get('item')}")
            print("-" * 50)
            print(f" 내용 요약:\n{latest_doc[:200]}...")
            print("=" * 50 + "\n")
        else:
            log.warning("DB에 저장된 문서가 없어 내용을 확인할 수 없습니다.")
