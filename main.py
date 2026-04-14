# ============================================================
# main.py  ―  Kanana 전용 금융 에이전트
# ============================================================
import torch
import operator
import re
import os
import textwrap
from datetime import datetime, timedelta
from typing import List, Annotated, TypedDict, Any, Literal
from dateutil.relativedelta import relativedelta

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
try:
    from langchain_tavily import TavilySearch as _TavilyTool
    _TAVILY_NEW = True
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults as _TavilyTool
    _TAVILY_NEW = False
from langchain_huggingface import HuggingFaceEmbeddings

# langchain_chroma 우선 사용
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# BM25 조건부 import — rank_bm25 미설치 시 None으로 처리
try:
    from langchain_community.retrievers import BM25Retriever
    _BM25_AVAILABLE = True
except ImportError:
    BM25Retriever = None
    _BM25_AVAILABLE = False

from config import Config
from model_loader import KananaModel
from logger_setting import get_logger

log = get_logger("MainAgent")

# ── 임베딩 모델 싱글턴 (매 노드마다 재로드 방지) ──────────────────────────────
_embeddings: HuggingFaceEmbeddings | None = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBED_MODEL_NAME,
            model_kwargs=Config.EMBED_MODEL_KWARGS,
            encode_kwargs=Config.EMBED_ENCODE_KWARGS,
        )
    return _embeddings


# ── 상태 정의 ──────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question: str
    topic: Literal["finance", "general", "off_topic"]

    category: str
    target_date: str
    target_date_int: int    # Chroma $gte/$lte용 YYYYMMDD int
    start_date_int: int     # 범위 검색 시작
    end_date_int: int       # 범위 검색 종료
    target_section: str
    multi_queries: List[str]

    retrieved_docs: List[Any]
    is_fallback: bool

    context_score: str          # "yes" / "no"
    hallucination_score: str    # "yes" / "no"

    answer: str
    analysis_note: str

    loop_count: Annotated[int, operator.add]
    retry_count: Annotated[int, operator.add]

    answer_score: str


# ── 공통 헬퍼: Kanana 호출 ──────────────────────────────────────────────────────
def ask_kanana(prompt: str, max_tokens: int = 1024, temp: float = 0.1) -> str:
    """Kanana 모델에 프롬프트를 전달하고 어시스턴트 응답만 반환합니다."""
    model, tokenizer = KananaModel.get_model()

    do_sample = temp > 0
    generate_kwargs: dict = {
        "max_new_tokens": max_tokens,
        "temperature": temp,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": 1.15,
    }
    if do_sample:
        generate_kwargs["top_p"] = 0.9

    formatted_prompt = f"<|user|>\n{prompt}<|assistant|>\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(Config.DEVICE)

    with torch.no_grad():
        try:
            outputs = model.generate(**inputs, **generate_kwargs)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 어시스턴트 응답 부분만 추출
            if "<|assistant|>\n" in decoded:
                result = decoded.split("<|assistant|>\n")[-1]
            elif "<|assistant|>" in decoded:
                result = decoded.split("<|assistant|>")[-1]
            else:
                plain_prompt = f"<|user|>\n{prompt}"
                result = decoded.replace(plain_prompt, "") if plain_prompt in decoded else decoded

            # result 내에 잔존하는 모든 역할 토큰 후처리로 제거
            if "<|user|>" in result:
                result = result.split("<|user|>")[0]

            # <|assistant|> 가 result 내부에 반복 등장하면 첫 번째 이전까지만 사용
            if "<|assistant|>" in result:
                result = result.split("<|assistant|>")[0]

            return result.strip()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.error("GPU OOM. 캐시를 비웁니다.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise

# ── Input Router ─────────────────────────────────────────────────────────
def input_router_node(state: AgentState):
    print("\n--- [NODE] Input Router ---")
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    question = state["question"]

    #LLM 호출 전 키워드로 선별
    _general_pre = [
        "시스템", "누구", "안녕", "뭐하는", "이건", "너는", "도와줘",
        "어떤 일", "무슨 일", "할 수 있", "뭘 할", "뭐 해줄", "어떻게 써",
        "소개", "기능", "사용법", "넌 뭐", "넌 어떤", "당신은 누구", "당신은 뭐",
        "가 뭐야", "이 뭐야", "란 뭐야", "는 뭐야", "가 뭐지", "이 뭐지",
        "란 무엇", "는 무엇", "뭐야", "뭐지", "이란", "이란 무엇", "무엇인가",
        "어떤 질문", "무슨 질문", "잘 답변", "응답 잘"
    ]
    # off_topic 사전 판단 키워드 — 감정 표현/욕설 또는 투자 조언 요청
    _off_topic_pre = [
        "짜증", "열받", "빡쳐", "ㅅㅂ", "ㅆㅂ", "씨발", "개새", "존나",
        "욕설", "꺼져", "닥쳐", "미치겠", "빡친", "화난다",
    ]
    # 투자 조언/추천 요청 — finance 키워드가 있어도 off_topic으로 처리
    _invest_advice = [
        "추천해줘", "추천해 줘", "추천 해줘", "뭐 사야", "뭘 사야",
        "사야 해", "사야해", "투자해도 돼", "투자 해도 돼",
        "종목 추천", "어디 투자", "어디에 투자",
    ]
    _finance_keywords = [
        "금리", "환율", "달러", "엔화", "유로", "주가", "NDF", "CDS",
        "금융", "채권", "주식", "원화", "증시", "코스피", "나스닥", "환", "이자",
        "관세", "트럼프", "연준", "기준금리", "국채", "펀드", "ETF", "선물",
    ]

    has_general_kw  = any(k in question for k in _general_pre)
    has_finance_kw  = any(k in question for k in _finance_keywords)
    has_off_topic   = any(k in question for k in _off_topic_pre)
    has_invest_advice = any(k in question for k in _invest_advice)

    # off_topic 사전 판단 — 감정/욕설 또는 투자 조언 요청이면 바로 off_topic 확정
    if has_off_topic or has_invest_advice:
        print(f">> topic=off_topic | requires_search=False (사전 감정/욕설 판단)")
        return {
            "topic": "off_topic",
            "is_fallback": True,
            "analysis_note": "off_topic 사전 판단",
            "retrieved_docs": [],
        }

    # general 사전 판단 — finance 키워드가 없으면서 general 키워드가 있으면 바로 확정
    if has_general_kw and not has_finance_kw:
        print(f">> topic=general | requires_search=False (사전 키워드 판단)")
        return {
            "topic": "general",
            "is_fallback": True,
            "analysis_note": "general 사전 키워드 판단",
            "retrieved_docs": [],
        }

    prompt = f"""현재 시각: {current_time_str}
당신은 국제 금융 정보 에이전트의 관문입니다. 아래 질문을 분류하세요.

분류 기준:
- finance: 금리/환율/주가/NDF/CDS/엔화/달러/유로/원화/증시/채권 등 데이터 검색이나 시황 분석이 필요한 질문
- general: "이건 무슨 시스템이야?", "안녕", "너는 누구니", "넌 어떤 일을 할 수 있어?" 등 에이전트/시스템 소개 및 단순 인사
- off_topic: 금융과 전혀 무관하거나 욕설, 정치 비난

질문: {question}

주의: 아래 세 줄만 출력하고 다른 말은 절대 하지 마세요.
TOPIC: finance 또는 general 또는 off_topic
SEARCH: True (finance인 경우만) 또는 False
REASON: 판단 근거 한 줄"""

    # 분류 판정만 하므로 max_tokens 최소화, temp=0.0으로 결정적 출력
    res = ask_kanana(prompt, max_tokens=60, temp=0.0)
    res_lower = res.lower()

    # topic 파싱
    if "topic: finance" in res_lower:
        topic = "finance"
    elif "topic: general" in res_lower:
        topic = "general"
    elif "topic: off_topic" in res_lower:
        topic = "off_topic"
    else:
        if has_general_kw:
            topic = "general"
        elif has_finance_kw:
            topic = "finance"
        else:
            topic = "general"
        log.warning(f"[input_router] topic 파싱 실패 → 키워드 기반 판단: topic={topic} | LLM 응답: {res!r}")

    if topic == "finance":
        requires_search = True
    else:
        requires_search = "search: true" in res_lower

    print(f">> topic={topic} | requires_search={requires_search}")
    return {
        "topic": topic,
        "is_fallback": not requires_search,
        "analysis_note": res,
        "retrieved_docs": [],
    }


def route_after_input(state: AgentState):
    topic = state.get("topic")
    is_fallback = state.get("is_fallback", False)
    if topic == "finance" and not is_fallback:
        return "finance_search"
    return "direct_answer"


# ── 상대 날짜 Python 계산 ─────────────────────────────────────────
_REL_DATE_PATTERNS = [
    (re.compile(r"(\d+)\s*일\s*전"),      lambda m, n: n - timedelta(days=int(m.group(1)))),
    (re.compile(r"(\d+)\s*주\s*전"),      lambda m, n: n - timedelta(weeks=int(m.group(1)))),
    (re.compile(r"(\d+)\s*개월\s*전"),    lambda m, n: n - relativedelta(months=int(m.group(1)))),
    (re.compile(r"(\d+)\s*달\s*전"),      lambda m, n: n - relativedelta(months=int(m.group(1)))),
    (re.compile(r"(\d+)\s*년\s*전"),      lambda m, n: n - relativedelta(years=int(m.group(1)))),
    (re.compile(r"어제"),                  lambda m, n: n - timedelta(days=1)),
    (re.compile(r"그저께|그제"),           lambda m, n: n - timedelta(days=2)),
    (re.compile(r"지난\s*주"),             lambda m, n: n - timedelta(weeks=1)),
    (re.compile(r"지난\s*달|저번\s*달"),   lambda m, n: n - relativedelta(months=1)),
    (re.compile(r"재작년"),                lambda m, n: n - relativedelta(years=2)),
    (re.compile(r"작년|전년"),             lambda m, n: n - relativedelta(years=1)),
]

def _extract_relative_date(question: str, now: datetime) -> str | None:
    """질문에서 상대 날짜를 계산해 'YYYY-MM-DD' 반환. 없으면 None."""
    for pattern, calc in _REL_DATE_PATTERNS:
        m = pattern.search(question)
        if m:
            return calc(m, now).strftime("%Y-%m-%d")
    return None

def _date_to_int(date_str: str) -> int:
    """'YYYY-MM-DD' → 20260403 (Chroma $gte/$lte 필터용 int)"""
    return int(date_str.replace("-", ""))


# ── Multi-Query Generator ────────────────────────────────────────────────
def multi_query_generator_node(state: AgentState):
    print("\n--- [NODE] Multi-Query Generator ---")
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    weekday_str = now.strftime("%A")
    time_str = now.strftime("%H:%M")
    question = state["question"]

    rel_date = _extract_relative_date(question, now)
    date_hint = (
        f"\n[날짜 힌트] 질문의 날짜 표현 계산 결과: {rel_date} → DATE에 이 값을 사용하세요."
        if rel_date else ""
    )

    prompt = f"""오늘 날짜: {today_str} ({weekday_str}), 현재 시각: {time_str}{date_hint}
당신은 전문 금융 검색어 추출기입니다. 사용자의 질문에서 핵심 키워드만 뽑아 검색용 쿼리 3개를 생성하세요.

[엄격한 규칙]
1. 사용자가 언급한 국가(미국, 한국, 일본 등)나 지표(금리, 환율 등)를 절대 임의로 바꾸지 마세요.
2. '최근', '어때', '알려줘' 등의 서술어는 제외하고 명사 위주로 검색어를 만드세요.
3. 08:00 이전이면 target_date를 전일로 설정하세요. (일요일·공휴일이면 가장 가까운 영업일로 조정)
4. 섹션 키워드: 주요뉴스, 뉴스, 동향 → 종합뉴스 / 국제금융시장, 금리, 환율, NDF → 금융지표_종합

질문: {question}

주의: 아래 형식 6줄만 출력하고 다른 말은 절대 하지 마세요.
DATE: YYYY-MM-DD
SECTION: 종합뉴스 또는 금융지표_종합 또는 전체
Q1: 검색어1
Q2: 검색어2
Q3: 검색어3
CATEGORY: 금리 또는 환율 또는 주식 또는 기타"""

    # 6줄 구조화 출력만 하므로 temp=0.0으로 안정적 출력, 쿼리 잘림 방지를 위해 충분한 토큰 확보
    res = ask_kanana(prompt, max_tokens=180, temp=0.0)
    
    # 응답 클리닝
    if "<|user|>" in res:
        res = res.split("<|user|>")[0]
    for stop_marker in ["주의:", "[시간 규칙]", "[엄격한 규칙]", "당신은 전문"]:
        if stop_marker in res:
            res = res.split(stop_marker)[0]
    res = res.strip()

    # 타겟 날짜 파싱
    if rel_date:
        target_date = rel_date
    else:
        date_match = re.search(r"DATE:\s*(\d{4}-\d{2}-\d{2})", res)
        target_date = date_match.group(1) if date_match else today_str

    # 타겟 섹션 파싱
    _VALID_SECTIONS = {"종합뉴스", "금융지표_종합", "전체"}
    section_match = re.search(r"SECTION:\s*(\S+)", res)
    raw_section = section_match.group(1).strip() if section_match else "전체"

    if raw_section not in _VALID_SECTIONS:
        if any(k in question for k in ["환율", "엔화", "달러", "유로", "금리", "NDF", "CDS"]):
            raw_section = "금융지표_종합"
        elif any(k in question for k in ["뉴스", "시황", "동향"]):
            raw_section = "종합뉴스"
        else:
            raw_section = "전체"
    target_section = raw_section

    # 쿼리 파싱
    _STOP_TOKENS = {"<|user|>", "<|assistant|>", "CATEGORY:", "DATE:", "SECTION:"}
    queries = []
    for i in range(1, 4):
        q_match = re.search(rf"Q{i}:\s*(.+)", res)
        if q_match:
            q_text = q_match.group(1).strip()
            for st in _STOP_TOKENS:
                if st in q_text:
                    q_text = q_text.split(st)[0].strip()
            if q_text:
                queries.append(q_text)

    if not queries:
        queries = [question, f"{target_date} 환율", f"{target_date} 금융시장"]

    # 카테고리 파싱
    _VALID_CATS = {"금리", "환율", "주식", "기타"}
    category = "금융"
    cat_match = re.search(r"CATEGORY:\s*(\S+)", res)
    if cat_match:
        raw_cat = cat_match.group(1).strip()
        raw_cat = re.split(r"[|/\s<]", raw_cat)[0].strip()
        if raw_cat in _VALID_CATS:
            category = raw_cat
        else:
            if any(k in question for k in ["환율", "엔화", "달러"]):
                category = "환율"
            elif any(k in question for k in ["금리", "금리차", "기준금리"]):
                category = "금리"
            elif any(k in question for k in ["주가", "증시", "코스피"]):
                category = "주식"
            else:
                category = "기타"

    # 날짜 범위 결정 로직 - 연도 범위 먼저 계산 후 나머지 조건 처리
    try:
        target_date_obj = datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        target_date_obj = datetime.strptime(today_str, "%Y-%m-%d")
        target_date = today_str

    today_obj = datetime.strptime(today_str, "%Y-%m-%d")

    # ── 연(年) 단위 범위 감지 ──────────────────────────────────────────────────
    # 절대 연도: 4자리(2024년), 2자리 약칭(25년) findall로 전체 추출
    # 두 연도 비교 - re.findall 여러 연도를 모두 추출해 min~max 범위 처리
    _abs_year_list  = [int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\s*년", question)]
    # 2자리 약칭: "25년"→2025 (이미 4자리로 파싱된 것과 중복 제거)
    _abbr_year_list = [2000 + int(y) for y in re.findall(r"(?<!\d)([2-9]\d)\s*년", question)
                       if (2000 + int(y)) not in _abs_year_list]
    _all_abs_years  = sorted(set(_abs_year_list + _abbr_year_list))

    # 상대 연도: 절대 연도가 없을 때만 적용, 1자리 숫자만 허용
    _rel_year_match = re.search(r"(?<!\d)([1-9])\s*년", question) if not _all_abs_years else None
    _kor_year_map   = {"일년": 1, "이년": 2, "삼년": 3, "사년": 4, "오년": 5}
    _kor_year_match = next((v for k, v in _kor_year_map.items() if k in question), None)
    _special_year   = (
        2 if "재작년" in question else
        1 if any(kw in question for kw in ["작년", "전년"]) else
        None
    )

    # 단일 절대 연도
    abs_year = _all_abs_years[0] if len(_all_abs_years) == 1 else None

    # 상대 연도 수 결정
    year_count = None
    if not _all_abs_years:
        if _rel_year_match:   year_count = int(_rel_year_match.group(1))
        elif _kor_year_match: year_count = _kor_year_match
        elif _special_year:   year_count = _special_year

    if len(_all_abs_years) >= 2:
        # ── 복수 절대 연도 비교: "25년이랑 26년", "2024년과 2025년" 등 ──────────
        # 가장 이른 연도 1월 1일 ~ 가장 늦은 연도 12월 31일로 범위 설정
        start_date_str = f"{min(_all_abs_years)}-01-01"
        end_date_str   = f"{max(_all_abs_years)}-12-31"
        target_date    = start_date_str

    elif abs_year:
        # ── 절대 연도: "2026년 3월 금리", "2025년 하반기 환율" 등 ───────────
        _month_map = {"1월":1,"2월":2,"3월":3,"4월":4,"5월":5,"6월":6,
                      "7월":7,"8월":8,"9월":9,"10월":10,"11월":11,"12월":12}
        _abs_month = next((v for k, v in _month_map.items() if k in question), None)
        _half      = (1,6)  if any(kw in question for kw in ["상반기","전반기"]) else \
                     (7,12) if any(kw in question for kw in ["하반기","후반기"]) else None
        _quarter   = None
        for qn,(qs,qe) in [("1분기",(1,3)),("2분기",(4,6)),("3분기",(7,9)),("4분기",(10,12))]:
            if qn in question: _quarter=(qs,qe); break

        if _abs_month:
            import calendar
            last_day = calendar.monthrange(abs_year, _abs_month)[1]
            start_date_str = f"{abs_year}-{_abs_month:02d}-01"
            end_date_str   = f"{abs_year}-{_abs_month:02d}-{last_day:02d}"
        elif _half:
            start_date_str = f"{abs_year}-{_half[0]:02d}-01"
            end_date_str   = f"{abs_year}-{_half[1]:02d}-{'30' if _half[1] in (6,9,11) else '31'}"
        elif _quarter:
            start_date_str = f"{abs_year}-{_quarter[0]:02d}-01"
            end_date_str   = f"{abs_year}-{_quarter[1]:02d}-{'30' if _quarter[1] in (6,9,11) else '31'}"
        else:
            start_date_str = f"{abs_year}-01-01"
            end_date_str   = f"{abs_year}-12-31"
        target_date = start_date_str

    elif year_count:
        base_year_obj = today_obj - relativedelta(years=year_count)
        base_year = base_year_obj.year

        _half = None
        if any(kw in question for kw in ["상반기", "전반기"]):
            _half = (1, 6)
        elif any(kw in question for kw in ["하반기", "후반기"]):
            _half = (7, 12)

        _quarter = None
        for qn, (qs, qe) in [("1분기",(1,3)),("2분기",(4,6)),
                               ("3분기",(7,9)),("4분기",(10,12))]:
            if qn in question:
                _quarter = (qs, qe); break

        if _half:
            start_date_str = f"{base_year}-{_half[0]:02d}-01"
            end_date_str   = f"{base_year}-{_half[1]:02d}-{'30' if _half[1] in (6,9,11) else '31'}"
            target_date    = start_date_str
        elif _quarter:
            start_date_str = f"{base_year}-{_quarter[0]:02d}-01"
            end_date_str   = f"{base_year}-{_quarter[1]:02d}-{'30' if _quarter[1] in (6,9,11) else '31'}"
            target_date    = start_date_str
        else:
            is_range = any(kw in question for kw in ["비교", "대비", "vs", "와 비교", "과 비교"])
            if is_range:
                end_date_str   = today_str
                start_date_str = base_year_obj.strftime("%Y-%m-%d")
                target_date    = today_str
            else:
                start_date_str = f"{base_year}-01-01"
                end_date_str   = f"{base_year}-12-31"
                target_date    = f"{base_year}-01-01"
    elif "어제" in question and any(kw in question for kw in ["그제", "그저께"]):
        start_date_str = (today_obj - timedelta(days=2)).strftime("%Y-%m-%d")
        end_date_str   = (today_obj - timedelta(days=1)).strftime("%Y-%m-%d")
        target_date    = end_date_str
    elif re.search(r"이번\s*주|금주", question):
        # "이번 주" / "이번주" / "금주" → 이번 주 월요일~오늘 범위
        days_since_monday = today_obj.weekday()  # 월=0, 일=6
        start_date_str = (today_obj - timedelta(days=days_since_monday)).strftime("%Y-%m-%d")
        end_date_str   = today_str
    elif any(kw in question for kw in ["달", "개월"]):
        start_date_str = (target_date_obj - relativedelta(months=1)).strftime("%Y-%m-%d")
        end_date_str = target_date
    elif any(kw in question for kw in ["최근", "요즘", "동향", "추이", "흐름"]):
        start_date_str = (target_date_obj - timedelta(days=7)).strftime("%Y-%m-%d")
        end_date_str = target_date
    else:
        # ── 연도 없는 복수 월 비교: "3월, 4월 엔화", "3월이랑 4월 금리" 등 ──────
        # 절대/상대 연도, 다른 기간 조건 없지만 월 키워드 2개 이상 -> 올해 해당 월 범위 처리
        import calendar as _cal
        _month_map2 = {"1월":1,"2월":2,"3월":3,"4월":4,"5월":5,"6월":6,
                       "7월":7,"8월":8,"9월":9,"10월":10,"11월":11,"12월":12}
        _found_months = sorted(set(v for k, v in _month_map2.items() if k in question))
        if len(_found_months) >= 2:
            _this_year = today_obj.year
            _m_start   = min(_found_months)
            _m_end     = max(_found_months)
            _last_day  = _cal.monthrange(_this_year, _m_end)[1]
            start_date_str = f"{_this_year}-{_m_start:02d}-01"
            end_date_str   = f"{_this_year}-{_m_end:02d}-{_last_day:02d}"
            target_date    = start_date_str
        else:
            start_date_str = target_date
            end_date_str   = target_date

    target_date_int = _date_to_int(target_date)
    start_date_int = _date_to_int(start_date_str)
    end_date_int   = _date_to_int(end_date_str)

    print(f">> target_date={target_date} | section={target_section} | category={category}")
    print(f">> queries={queries}")
    print(f">> date_range={start_date_str} ~ {end_date_str}")

    return {
        "target_date":     target_date,
        "target_date_int": target_date_int,
        "start_date_int":  start_date_int,
        "end_date_int":    end_date_int,
        "target_section":  target_section,
        "multi_queries":   queries,
        "category":        category,
    }

# vector_db는 서버 시작 시 startup()에서 초기화
vector_db = None

def ensure_date_int_metadata():
    """
    기존 Chroma DB 문서에 date_int 필드가 없으면 자동으로 추가합니다.
    date_int는 날짜 범위 검색($gte/$lte)을 위해 필요한 정수형 메타데이터입니다.
    최초 1회 실행 후에는 변경 사항이 없으면 빠르게 종료됩니다.

    사용법: 스크립트 시작 시 한 번만 호출하세요.
    예) ensure_date_int_metadata()
    """
    try:
        all_docs = vector_db.get(include=["metadatas"])
        if not all_docs or not all_docs.get("ids"):
            return
        
        ids_to_update, metas_to_update = [], []
        for doc_id, meta in zip(all_docs["ids"], all_docs["metadatas"]):
            if meta and "date" in meta and "date_int" not in meta:
                try:
                    date_int = int(meta["date"].replace("-", ""))
                    new_meta = {**meta, "date_int": date_int}
                    ids_to_update.append(doc_id)
                    metas_to_update.append(new_meta)
                except (ValueError, AttributeError):
                    continue
        
        if ids_to_update:
            # Chroma LangChain 래퍼는 .update() 미지원 → _collection 직접 접근
            BATCH = 500
            for i in range(0, len(ids_to_update), BATCH):
                vector_db._collection.update(
                    ids=ids_to_update[i:i+BATCH],
                    metadatas=metas_to_update[i:i+BATCH],
                )
            log.info(f"[date_int 마이그레이션] {len(ids_to_update)}건 완료.")
        else:
            log.info("[date_int 마이그레이션] 이미 최신 상태입니다.")
    except Exception as e:
        log.warning(f"[date_int 마이그레이션 실패] {e} — 문자열 후처리 방식으로 폴백합니다.")

def get_target_item(target_section: str) -> str | None:
    """중복되는 target_item 매핑 로직을 함수로 분리"""
    if target_section in ("주요뉴스", "종합뉴스"):
        return "종합뉴스"
    elif target_section in ("국제금융시장", "금융지표_종합"):
        return "금융지표_종합"
    return None

# ── Check Availability ───────────────────────────────────────────────────
def check_availability_node(state: AgentState):
    log.info("\n--- [NODE] Check Availability ---")
    target_date    = state.get("target_date", "")
    start_date_int = state.get("start_date_int", 0)
    end_date_int   = state.get("end_date_int", 0)
    target_item    = get_target_item(state.get("target_section"))

    # ── 단일 날짜 존재 여부 확인 ──
    def _check_single(date_str: str) -> bool:
        try:
            f = {"date": {"$eq": date_str}}
            if target_item:
                f = {"$and": [{"date": {"$eq": date_str}}, {"item": {"$eq": target_item}}]}
            db_res = vector_db.get(where=f, limit=1)
            return bool(db_res and db_res.get("ids"))
        except Exception:
            return False

    has_data = False
    try:
        # 1) target_date 기준 단일 날짜 확인
        has_data = _check_single(target_date)

        # 2) 범위 쿼리인 경우(start != end) → 여러 샘플 날짜로 DB 보유 여부 판단
        if not has_data and start_date_int and end_date_int and start_date_int != end_date_int:
            from datetime import date as _date
            start_obj = datetime.strptime(str(start_date_int), "%Y%m%d")
            end_obj   = datetime.strptime(str(end_date_int),   "%Y%m%d")
            delta_days = (end_obj - start_obj).days

            # 최대 10개 샘플 날짜로 존재 여부 확인 (start, end, 균등 분할)
            sample_dates = {start_obj, end_obj}
            steps = min(8, delta_days)
            if steps > 0:
                for i in range(1, steps + 1):
                    sample_dates.add(start_obj + timedelta(days=int(delta_days * i / (steps + 1))))

            for sd in sorted(sample_dates):
                if _check_single(sd.strftime("%Y-%m-%d")):
                    has_data = True
                    break

        log.info(f">> [{'성공' if has_data else '미보유'}] 데이터 존재 여부 확인.")
        return {"is_fallback": not has_data}

    except Exception as e:
        log.error(f"Check Availability 오류: {e}")
        return {"is_fallback": True}


# ── 헬퍼: Chroma 날짜 필터 생성 ─────────────────────────────────────────────────
def _build_date_filter(start_date_int: int, end_date_int: int, target_item: str | None) -> dict | None:
    """
    Chroma는 숫자형 메타데이터에만 $gte/$lte를 지원합니다.
    date_int (예: 20260408) 필드를 사용해 범위 필터를 구성합니다.
    date_int 필드가 DB에 없을 경우를 대비해 $eq 단일 날짜 폴백도 지원합니다.
    """
    filter_list = []

    if start_date_int and end_date_int:
        if start_date_int == end_date_int:
            # 단일 날짜: int 필드와 문자열 필드 둘 다 시도할 수 있도록 $or 사용
            date_str = datetime.strptime(str(start_date_int), "%Y%m%d").strftime("%Y-%m-%d")
            filter_list.append({
                "$or": [
                    {"date_int": {"$eq": start_date_int}},
                    {"date": {"$eq": date_str}},
                ]
            })
        else:
            # 문자열 $gte/$lte는 Chroma에서 오류 발생 -> 제거
            filter_list.append({
                "$and": [
                    {"date_int": {"$gte": start_date_int}},
                    {"date_int": {"$lte": end_date_int}},
                ]
            })

    if target_item:
        filter_list.append({"item": {"$eq": target_item}})

    if not filter_list:
        return None
    if len(filter_list) == 1:
        return filter_list[0]
    return {"$and": filter_list}


def _search_with_fallback(query: str, start_date_int: int, end_date_int: int,
                          target_item: str | None, k: int = 5) -> list:
    """
    1차: 날짜 int + item 필터 검색
    2차: item 필터만으로 재시도 (날짜 필터 오류 시)
    3차: 필터 없이 전체 검색 후 날짜 문자열로 후처리 필터링
    """
    start_str = datetime.strptime(str(start_date_int), "%Y%m%d").strftime("%Y-%m-%d") if start_date_int else None
    end_str   = datetime.strptime(str(end_date_int),   "%Y%m%d").strftime("%Y-%m-%d") if end_date_int   else None

    # ── 1차 시도: 정상 필터 ──
    try:
        f = _build_date_filter(start_date_int, end_date_int, target_item)
        docs = vector_db.similarity_search(query, k=k, filter=f)
        if docs:
            return docs
    except Exception as e:
        log.warning(f"[RAG 1차 필터 실패] {e}")

    # ── 2차 시도: item 필터만 ──
    try:
        f2 = {"item": {"$eq": target_item}} if target_item else None
        docs2 = vector_db.similarity_search(query, k=k * 2, filter=f2)
        if docs2 and start_str and end_str:
            # 날짜 문자열로 후처리 필터링
            docs2 = [d for d in docs2
                     if start_str <= d.metadata.get("date", "") <= end_str]
        if docs2:
            log.info("[RAG 2차 item-only 필터 성공]")
            return docs2[:k]
    except Exception as e:
        log.warning(f"[RAG 2차 필터 실패] {e}")

    # ── 3차 시도: 필터 없이 + 날짜 문자열 후처리 ──
    # date_int 마이그레이션 전 또는 범위가 넓을 때 활용
    try:
        # 범위가 넓을수록 k 동적 확대
        k3_mult = 6 if (start_str and end_str and start_str != end_str) else 3
        docs3 = vector_db.similarity_search(query, k=k * k3_mult)
        if docs3 and start_str and end_str:
            docs3 = [d for d in docs3
                     if start_str <= d.metadata.get("date", "") <= end_str]
        if docs3:
            log.info("[RAG 3차 필터 없이 + 후처리 성공]")
            return docs3[:k]
    except Exception as e:
        log.warning(f"[RAG 3차 전체 검색 실패] {e}")

    # ── 4차 시도: DB 전체 스캔 후 날짜 문자열 후처리 ──
    if start_str and end_str and start_str != end_str:
        try:
            all_in_range = vector_db.get(include=["documents", "metadatas"])
            if all_in_range and all_in_range.get("ids"):
                matched = []
                for doc_content, meta in zip(all_in_range["documents"], all_in_range["metadatas"]):
                    doc_date = (meta or {}).get("date", "")
                    if start_str <= doc_date <= end_str:
                        if not target_item or (meta or {}).get("item") == target_item:
                            matched.append(Document(page_content=doc_content, metadata=meta or {}))
                if matched:
                    log.info(f"[RAG 4차 전체스캔 성공] 범위({start_str}~{end_str}) 내 {len(matched)}건 발견")
                    return matched[:k]
        except Exception as e:
            log.warning(f"[RAG 4차 전체 스캔 실패] {e}")

    return []


# ── RAG Searcher ─────────────────────────────────────────────────────────
def rag_searcher_node(state: AgentState):
    log.info("\n--- [NODE] RAG Searcher ---")
    try:
        queries      = state.get("multi_queries", [state["question"]])
        target_item  = get_target_item(state.get("target_section"))
        start_date_int = state.get("start_date_int", 0)
        end_date_int   = state.get("end_date_int",   0)

        start_date_str = datetime.strptime(str(start_date_int), "%Y%m%d").strftime("%Y-%m-%d") if start_date_int else "N/A"
        end_date_str   = datetime.strptime(str(end_date_int),   "%Y%m%d").strftime("%Y-%m-%d") if end_date_int   else "N/A"

        candidate_docs = []
        for query in queries:
            docs = _search_with_fallback(query, start_date_int, end_date_int, target_item, k=5)
            candidate_docs.extend(docs)

        # 중복 문서 제거
        unique_candidates = list({d.page_content: d for d in candidate_docs}.values())

        if not unique_candidates:
            log.warning(f"검색 결과 없음 (기간: {start_date_str}~{end_date_str}). 웹 검색으로 전환.")
            return {"retrieved_docs": [], "is_fallback": True}

        log.info(f">> 기간({start_date_str}~{end_date_str}) 내 최종 검색 문서: {len(unique_candidates)}건")
        return {"retrieved_docs": unique_candidates, "is_fallback": False}

    except Exception as e:
        log.error(f"RAG Searcher 전체 오류: {e}")
        return {"retrieved_docs": [], "is_fallback": True}


# ── Web Searcher ─────────────────────────────────────────────────────────
def web_searcher_node(state: AgentState):
    log.info("\n--- [NODE] Web Searcher ---")
    if not os.environ.get("TAVILY_API_KEY"):
        log.warning("TAVILY_API_KEY가 없습니다. 기존 문서를 반환합니다.")
        return {"retrieved_docs": state.get("retrieved_docs", []), "loop_count": 1}
    
    if _TAVILY_NEW:
        web_tool = _TavilyTool(max_results=3, search_depth="advanced")
    else:
        web_tool = _TavilyTool(k=3, search_depth="advanced")
    
    query_parts = [state.get("target_date", ""), state.get("category", ""), state.get("question", "")]
    search_query = " ".join([p for p in query_parts if p]).strip()
    
    log.info(f">> Web Searching: {search_query}")

    web_docs = []
    # 웹 문서 오염 콘텐츠 사전 필터 패턴 (LaTeX 블록/인라인, HTML 블록 태그, 학술 논문 본문)
    _WEB_NOISE_RE = re.compile(
        r"\\section\{|\\subsection\{|\\begin\{|\\cite\{|\\label\{|"
        r"\\ref\{|\\textbf\{|\\emph\{|"
        r"\\\(\\mathrm|\\frac\{|\\sqrt\{|"          # 인라인 수식 명령어
        r"<h[1-6][\s>]|<p[\s>]|<div[\s>]|<table[\s>]",  # HTML 블록 태그
        re.IGNORECASE
    )
    try:
        raw = web_tool.invoke({"query": search_query})

        # dict 형태: {"results": [...]}
        if isinstance(raw, dict):
            result_list = raw.get("results", [])
        elif isinstance(raw, list):
            result_list = raw
        elif isinstance(raw, str):
            result_list = []
            if raw.strip():
                doc = Document(
                    page_content=f"[Web] {search_query}\n{raw[:1600]}",
                    metadata={"date": state.get("target_date"), "item": "웹데이터", "url": "Tavily Search"},
                )
                web_docs.append(doc)
                log.info(f"[Web Searcher] 문자열 응답 처리 완료 ({len(raw)}자)")
        else:
            result_list = []
            log.warning(f"[Web Searcher] 알 수 없는 응답 형태: {type(raw)}")

        for r in result_list:
            if not isinstance(r, dict):
                continue
            title   = r.get('title', '') or ''
            content = r.get('content', '') or ''

            if _WEB_NOISE_RE.search(content):
                log.warning(f"[Web Sanitize] LaTeX 오염 문서 제외: {title[:60]}")
                continue

            content_preview = content[:800]
            doc = Document(
                page_content=f"[Web] {title}\n{content_preview}",
                metadata={
                    "date": state.get("target_date"),
                    "item": "웹데이터",
                    "url": r.get('url', '출처 없음'),
                }
            )
            web_docs.append(doc)
        log.info(f"[Web Searcher] 총 {len(web_docs)}개 문서 수집 완료")
    except Exception as e:
        log.error(f"Web Searcher 오류: {e}")

    final_docs = state.get("retrieved_docs", []) + web_docs
    
    return {"retrieved_docs": final_docs, "loop_count": 1}


# ── Context Filter ───────────────────────────────────────────────────────
def context_filter_node(state: AgentState):
    print("\n--- [NODE] Context Filter ---")
    question = state["question"]
    documents = state.get("retrieved_docs", [])
    target_date = state.get("target_date", "")
    category = state.get("category", "")
    loop_count = state.get("loop_count", 0)

    if not documents:
        print(">> 검색된 문서 없음.")
        return {"retrieved_docs": [], "context_score": "no"}

    # 문서가 1개이거나, 웹 검색한 경우 2~3개 문서도 LLM 필터링 생략 (데이터 손실 방지)
    if len(documents) <= 3 or loop_count >= 1:
        print(f">> [Skip] 검색된 문서가 {len(documents)}개이거나 웹 검색 데이터입니다. 안전을 위해 필터링을 통과시킵니다.")
        return {"retrieved_docs": documents, "context_score": "yes"}

    # 범위 쿼리 결과(날짜가 다양한 문서 집합)는 필터링 없이 통과
    doc_dates = set(d.metadata.get("date", "") for d in documents)
    if len(doc_dates) >= 3:
        print(f">> [Skip] 다중 날짜({len(doc_dates)}개) 범위 쿼리 결과 — 필터링 생략, 전체 통과.")
        return {"retrieved_docs": documents, "context_score": "yes"}

    docs_text = "\n\n".join([
        f"[문서 ID: {i}]\n{doc.page_content[:400]}" 
        for i, doc in enumerate(documents)
    ])

    system_prompt = (
        f"당신은 국제 금융 정보의 정밀 판독관입니다.\n"
        f"질문: {question}\n"
        f"타겟 날짜: {target_date} | 카테고리: {category}\n\n"
        f"아래 문서들 중 질문에 답할 수 있는 유의미한 수치나 분석이 포함된 문서의 ID만 골라내세요.\n"
        f"- 유효한 문서가 있다면 해당 ID를 쉼표로 구분해 출력하세요 (예: 0, 2, 3).\n"
        f"- 유효한 문서가 전혀 없다면 'NONE'이라고 출력하세요.\n\n"
        f"{docs_text}\n\n"
        f"유효한 문서 ID:"
    )

    raw_res = ask_kanana(system_prompt, max_tokens=20)
    filtered_docs = []
    
    is_empty_response = "NONE" in raw_res.upper() or "없" in raw_res

    if not is_empty_response:
        id_matches = re.findall(r"\d+", raw_res)
        valid_ids = set([int(x) for x in id_matches if int(x) < len(documents)])
        
        for i, doc in enumerate(documents):
            if i in valid_ids:
                filtered_docs.append(doc)
                print(f">> [Match]    [{doc.metadata.get('date', 'unknown')}] {doc.metadata.get('item', '지표')}")
            else:
                print(f">> [Filtered] [{doc.metadata.get('date', 'unknown')}] {doc.metadata.get('item', '지표')}")

    if not filtered_docs:
        if is_empty_response:
            print(f">> [Alert] 유효한 문서 없음 (LLM 응답: {raw_res!r})")
            return {"retrieved_docs": [], "context_score": "no"}
        else:
            print(f">> [Warning] LLM 파싱 오류({raw_res!r}). 문서를 유지합니다.")
            filtered_docs = documents

    print(f">> Filtering 완료: {len(documents)}개 → {len(filtered_docs)}개")
    return {"retrieved_docs": filtered_docs, "context_score": "yes"}


# ── Context Reranker ─────────────────────────────────────────────────────
def context_reranker_node(state: AgentState):
    print("\n--- [NODE] Context Reranker ---")
    question = state["question"]
    documents = state.get("retrieved_docs", [])
    target_date = state.get("target_date", "")
    category = state.get("category", "")

    if not documents or len(documents) == 1:
        print(">> 재정렬할 문서가 없거나 1개뿐입니다.")
        return {"retrieved_docs": documents}

    context_text = "\n\n".join([
        f"ID: {i} | [섹션: {doc.metadata.get('section','미분류')}] "
        f"[항목: {doc.metadata.get('item','지표')}] [날짜: {doc.metadata.get('date','unknown')}]\n"
        f"내용: {doc.page_content[:400].replace('{','[').replace('}',']')}"
        for i, doc in enumerate(documents)
    ])

    prompt = (
        f"당신은 금융 문서 재정렬 전문가입니다.\n"
        f"오늘 날짜: {target_date} | 카테고리: {category}\n"
        f"질문: {question}\n\n"
        f"아래 문서들 중 질문에 가장 관련성 높은 순으로 상위 5개의 ID를 쉼표로 나열하세요.\n"
        f"예시 출력: 2,0,4,1,3\n\n"
        f"{context_text}\n\n"
        f"상위 ID(쉼표 구분, 숫자만):"
    )

    res = ask_kanana(prompt, max_tokens=20, temp=0.0)
    id_matches = re.findall(r"\d+", res)

    if id_matches:
        selected_ids = []
        for x in id_matches:
            idx = int(x)
            # 인덱스 범위 초과 및 중복 추가 방지
            if idx < len(documents) and idx not in selected_ids:
                selected_ids.append(idx)
                
        reranked = [documents[i] for i in selected_ids]
        remaining = [d for i, d in enumerate(documents) if i not in selected_ids]
        top_docs = (reranked + remaining)[:5]
    else:
        top_docs = documents[:5]

    print(f">> 재정렬 완료: {len(top_docs)}개 선정.")
    return {"retrieved_docs": top_docs}


# ── Context Evaluator ────────────────────────────────────────────────────
def context_evaluator_node(state: AgentState):
    print("\n--- [NODE] Context Evaluator ---")
    question = state["question"]
    target_date = state.get("target_date", "")
    docs = state.get("retrieved_docs", [])
    loop_count = state.get("loop_count", 0)

    if not docs:
        print(">> 문서 없음. 웹 검색 경로 활성화 (또는 답변 생성 진입).")
        return {"is_fallback": True, "context_score": "no"}

    print(f">> {len(docs)}개 문서 평가 중... (loop_count={loop_count})")
    doc_summary = "\n".join([f"- {d.page_content[:200]}..." for d in docs[:3]])

    prompt = f"""당신은 금융 정보 완성도 평가관입니다.
아래 문서들이 질문에 답하기 위한 구체적 수치(금리, 환율, 지수)와 분석을 포함하면 YES, 그렇지 않으면 NO라고만 답하세요.

판단 기준:
1. 핵심 수치가 하나라도 명시되어 있으면 YES
2. 여러 문서를 조합해 흐름 설명이 가능하면 YES
3. 타겟 날짜({target_date}) 정보를 설명할 수 있으면 YES
4. 수치가 전혀 없거나 무관한 내용이면 NO

질문: {question}

[문서 요약]:
{doc_summary}

답변(YES 또는 NO):"""

    raw_res = ask_kanana(prompt, max_tokens=5, temp=0.0)
    yes_no_match = re.search(r"\b(YES|NO)\b", raw_res.strip().upper())
    
    if yes_no_match:
        first_token = yes_no_match.group(1)
    else:
        first_token = "YES" if raw_res.strip().upper().startswith("Y") else "NO"
        log.warning(f"[context_evaluator] YES/NO 파싱 실패: {raw_res!r} → {first_token}")

    is_enough = (first_token == "YES")
    log.info(f"결과: {'충분' if is_enough else '부족'} | 판정: {first_token!r}")

    if not is_enough and (len(docs) >= 3 or loop_count >= 1):
        log.warning("정보 불충분 판정이나 루프/문서 수 조건으로 답변 단계 진입.")
        is_enough = True

    return {
        "is_fallback": not is_enough,
        "context_score": "yes" if is_enough else "no",
    }


# ── 라우터 로직 ─────────────────────────────────────────
def route_from_evaluator(state: AgentState):
    score = state.get("context_score", "no")
    loop_count = state.get("loop_count", 0)
    
    if score == "yes" or loop_count >= 1:
        print(f"--- [ROUTE] 정보 충분 (loop={loop_count}) → 답변 생성 ---")
        return "Enough"
        
    print("--- [ROUTE] 정보 부족 → 웹 검색 보강 ---")
    return "Not_Enough"


def _clean_answer(text: str) -> str:
    """
    후처리 통합 버전: LaTeX/HTML 오염 차단, 면책 주석 제거, 중복 블록 제거,
    플레이스홀더 제거, 미래 날짜 참고문헌 제거, 미인용 참고문헌 제거
    """
    from datetime import date as _date_today

    _LATEX_MARKERS = [
        r"\\section\{", r"\\subsection\{", r"\\begin\{",
        r"\\cite\{", r"\\label\{", r"\\ref\{",
        r"\\textbf\{", r"\\emph\{",
        r"\\\(", r"\\\[",
        r"\\mathrm\{", r"\\frac\{", r"\\sqrt\{",
    ]
    _HTML_MARKERS = [
        r"<h[1-6][\s>]", r"</h[1-6]>",
        r"<p[\s>]", r"</p>",
        r"<div[\s>]", r"</div>",
        r"<table[\s>]", r"<tr[\s>]", r"<td[\s>]",
        r"<ul[\s>]", r"<ol[\s>]", r"<li[\s>]",
        r"</assistant>", r"<assistant>",
        r"<hex>", r"</hex>",
        r"<\|assistant\|>", r"<\|user\|>",
        r"<\|prompt\|>", r"<\|system\|>",
    ]
    _contam_re = re.compile("|".join(_LATEX_MARKERS + _HTML_MARKERS), re.IGNORECASE)

    lines = text.split("\n")
    cleaned = []
    seen_lines = set()
    in_ref_section = False
    ref_section_line_count = 0
    _REF_MAX_LINES = 6

    for line in lines:
        stripped = line.strip()

        if "]]]" in stripped or "[[[" in stripped or re.search(r"={5,}", stripped):
            break
        if _contam_re.search(stripped):
            break
        if stripped.startswith("※") or stripped.startswith("(※") or stripped.startswith("* ※"):
            continue

        if re.search(r"^\[참고\s*문헌\]|^참고\s*문헌[:\s]", stripped):
            in_ref_section = True
            ref_section_line_count = 0

        if in_ref_section and stripped:
            ref_section_line_count += 1
            if ref_section_line_count > _REF_MAX_LINES:
                break
            if ref_section_line_count > 1 and len(stripped) > 120:
                cleaned.append(stripped[:120])
                continue

        if not in_ref_section and stripped and len(stripped) >= 30:
            if not re.search(r'\d', stripped):
                pure_text = re.sub(r'^\[[^\]]+\]\s*:\s*', '', stripped)
                pure_text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', pure_text).strip()
                if pure_text and pure_text in seen_lines:
                    break
                if pure_text:
                    seen_lines.add(pure_text)

        cleaned.append(line)

    final_text = "\n".join(cleaned).strip()
    final_text = re.sub(r'\n\s*\n', '\n\n', final_text)

    # 역할 토큰 제거
    for stop_token in ["<|assistant|>", "<|user|>", "<|prompt|>", "<|system|>"]:
        if stop_token in final_text:
            final_text = final_text.split(stop_token)[0].strip()

    # 상세 분석 내부 인라인 참고문헌 제거
    ref_sec_start = re.search(r'(^\[참고\s*문헌\]|^참고\s*문헌[:\s])', final_text, re.MULTILINE)
    ref_sec_pos = ref_sec_start.start() if ref_sec_start else len(final_text)
    body = final_text[:ref_sec_pos]
    tail = final_text[ref_sec_pos:]
    body = re.sub(r'^\s*\[\d+\]\s+\d{4}-\d{2}-\d{2}[^\n]*\n?', '', body, flags=re.MULTILINE)
    final_text = body + tail

    # 중복 [참고 문헌] 블록 제거
    ref_matches = list(re.compile(r'(^\[참고\s*문헌\]|^참고\s*문헌[:\s])', re.MULTILINE).finditer(final_text))
    if len(ref_matches) >= 2:
        final_text = final_text[:ref_matches[1].start()].strip()

    # 중복 [요약] 블록 제거
    summary_matches = list(re.compile(r'(^\[요약\])', re.MULTILINE).finditer(final_text))
    if len(summary_matches) >= 2:
        final_text = final_text[:summary_matches[1].start()].strip()

    # 플레이스홀더 제거
    _placeholder_re = re.compile(
        r'^\s*(\[번호\]|\[숫자\])\s.*$'
        r'|^\s*\[\d+\]\s*날짜\s*\|.*$'
        r'|^\s*\[\d+\]\s*날짜\s*$'
        r'|^\s*날짜\s*\|\s*제목.*$'
        r'|^\s*\(최대\s*\d+개.*\).*$'
        r'|^\s*\[참고\s*문헌\]\.\.\.$$'
        r'|^\s*\[\d+\]\s*.*(기타\s*참고\s*자료|선택\s*사항).*$'
        r'|^\s*\[\d+\]\s*.*없음\(최대.*$'
        r'|^\s*\[\d+\]\s*.*—>.*없음.*$'
        r'|참고\s*자료\(\[번호\]\s*날짜\s*\|\s*제목\).*$'
        r'|^\s*\(총\s*\d+개\s*참고\s*문헌.*\).*$'
        r'|^\s*\[주석.*\].*$'
        r'|^\s*\[수정\s*내역\].*$'
        r'|^\s*\[교정\s*피드백\].*$',
        re.MULTILINE
    )
    final_text = _placeholder_re.sub('', final_text)

    # Document N / 문서 N → [N] 통일
    final_text = re.sub(r'\[?[Dd]ocument\s*#?\s*(\d+)\]?', r'[\1]', final_text)
    final_text = re.sub(r'\[?문서\s*#?\s*(\d+)\]?', r'[\1]', final_text)

    # 미래 날짜 참고문헌 제거 + 구분선 제거
    today_str = _date_today.today().strftime("%Y-%m-%d")
    ref_sec_match = re.search(r'(^\[참고\s*문헌\].*$)', final_text, re.MULTILINE)
    if ref_sec_match:
        body = final_text[:ref_sec_match.start()]
        ref_section = final_text[ref_sec_match.start():]
        filtered = []
        for line in ref_section.split("\n"):
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
            if date_match and re.match(r'^\s*\[\d+\]', line) and date_match.group(1) > today_str:
                continue
            if line.strip() == "---":
                continue
            filtered.append(line)
        final_text = body + "\n".join(filtered)

    # 미인용 참고문헌 제거
    ref_sec_match2 = re.search(r'(^\[참고\s*문헌\].*$)', final_text, re.MULTILINE)
    if ref_sec_match2:
        body = final_text[:ref_sec_match2.start()]
        ref_section = final_text[ref_sec_match2.start():]
        cited = set(re.findall(r'\[(\d+)\]', body))
        filtered = []
        for line in ref_section.split("\n"):
            ref_num = re.match(r'^\s*\[(\d+)\]', line)
            if ref_num and ref_num.group(1) not in cited:
                continue
            filtered.append(line)
        final_text = body + "\n".join(filtered)

    # 본문 없이 참고문헌만 남은 경우 원본 반환
    body_check = re.sub(r'(^|\n)\[참고\s*문헌\][\s\S]*', '', final_text, flags=re.MULTILINE).strip()
    if not body_check:
        return text.strip()

    # 빈 참고문헌 섹션 제거
    final_text = re.sub(r'\n\[참고\s*문헌\][:\s]*\n(\s*\n)*$', '', final_text, flags=re.MULTILINE).strip()

    final_text = re.sub(r'\n\s*\n', '\n\n', final_text).strip()
    return final_text


# ── Answer Generator ─────────────────────────────────────────────────────
def answer_generator_node(state: AgentState):
    print("\n--- [NODE] Answer Generator ---")
    topic = state.get("topic")

    if topic == "off_topic":
        # 감정 표현 + 금융 키워드가 섞인 경우 공감하며 금융 주제로 안내
        _finance_kw_in_q = any(k in state.get("question","") for k in [
            "금리","환율","달러","주가","증시","채권","금융","코스피","나스닥"
        ])
        if _finance_kw_in_q:
            return {"answer": "힘드셨군요. 금융 시장 관련 데이터나 분석이 필요하시면 편하게 질문해 주세요!"}
        return {"answer": "저는 국제 금융 및 외환 시장 분석 전문 에이전트입니다. 금융 관련 질문을 도와드릴게요."}
    if topic == "general":
        question = state.get("question", "")
        # 금융 용어 개념 질문: 간단히 설명
        _term_map = {
            "NDF": "NDF(Non-Deliverable Forward)는 실물 통화 교환 없이 차액만 정산하는 선물환 계약으로, 원화처럼 해외에서 거래가 제한된 통화의 환율 리스크를 헤지하는 데 사용됩니다.",
            "CDS": "CDS(Credit Default Swap)는 채권 발행자의 부도 위험을 거래하는 파생상품입니다. CDS 프리미엄이 높을수록 해당 국가나 기업의 신용 위험이 크다고 봅니다.",
            "ETF": "ETF(Exchange Traded Fund)는 주식처럼 거래소에서 거래되는 펀드로, 특정 지수나 자산을 추종합니다.",
            "기준금리": "기준금리는 중앙은행이 설정하는 정책금리로, 시중 금리의 기준이 됩니다. 한국은 한국은행, 미국은 연방준비제도(Fed)가 결정합니다.",
        }
        for term, explanation in _term_map.items():
            if term in question:
                return {"answer": explanation}
        return {"answer": "안녕하세요! 국제 금융 지표, 환율 전망, 시장 분석에 대해 궁금한 점을 물어봐 주세요."}

    question = state["question"]
    context_docs = state.get("retrieved_docs", [])
    target_date = state.get("target_date", "")

    if not context_docs:
        print(">> [Error] 활용 가능한 컨텍스트 없음.")
        return {"answer": "분석에 필요한 충분한 금융 데이터를 확보하지 못했습니다."}

    # 범위 쿼리 여부 판단 (start != end)
    start_date_int = state.get("start_date_int", 0)
    end_date_int   = state.get("end_date_int", 0)
    is_range_query = (start_date_int and end_date_int and start_date_int != end_date_int)
    # 범위 쿼리면 문서당 더 많은 내용 포함
    max_chars_per_doc = 1200 if is_range_query else 900

    formatted_context = ""
    for i, doc in enumerate(context_docs):
        d_date    = doc.metadata.get('date', '날짜 미상')
        d_item    = doc.metadata.get('item', '금융지표')
        d_section = doc.metadata.get('section', '미분류')
        d_url     = doc.metadata.get('url', '')
        is_web    = (d_item == "웹데이터")
        content_preview = doc.page_content[:max_chars_per_doc]
        source_label = f"[웹] {d_url}" if is_web else f"{d_section} ({d_item})"
        formatted_context += (
            f"--- 문서 [{i+1}] {'[웹문서]' if is_web else '[DB문서]'} ---\n"
            f"출처: {source_label} | 날짜: {d_date}\n"
            f"내용: {content_preview}\n\n"
        )

    # 프롬프트 제약조건 초강화, 범위 쿼리일 경우 비교/추이 분석 요청
    range_hint = ""
    if is_range_query:
        start_str_hint = str(start_date_int)
        end_str_hint   = str(end_date_int)
        try:
            start_str_hint = datetime.strptime(str(start_date_int), "%Y%m%d").strftime("%Y-%m-%d")
            end_str_hint   = datetime.strptime(str(end_date_int),   "%Y%m%d").strftime("%Y-%m-%d")
        except Exception:
            pass
        range_hint = f"\n[범위 분석 요청] 제공된 문서들이 {start_str_hint} ~ {end_str_hint} 기간의 데이터입니다. 날짜별로 변화 추이와 비교를 명확히 작성하세요."

    prompt = f"""당신은 전문 금융 분석가입니다. 오늘 날짜는 {target_date}입니다.{range_hint}
반드시 아래 제공된 [금융 데이터]만을 바탕으로 사용자의 질문에 답하세요.

[엄격한 금지사항]
1. 제공된 [금융 데이터]에 질문과 관련된 내용이 전혀 없다면, 절대로 지어내지 말고 "제공된 데이터에서 관련 정보를 찾을 수 없습니다."라고만 출력하고 답변을 즉시 마치세요.
2. 금융과 무관한 내용은 절대 출력하지 마세요.
3. 문서에 없는 수치나 원인을 배경지식으로 지어내지 마세요.
4. [참고 문헌] 섹션은 반드시 "[번호] 날짜 | 제목" 형식으로 한 줄씩, 최대 5개만 작성하고 즉시 멈추세요.
5. 답변이 끝나면 즉시 멈추세요. 주석, 면책 문구, 반복 설명을 절대 붙이지 마세요.

[답변 구조]
[요약]: 핵심 결론 2~3문장
[상세 분석]: 구체적 수치, 출처 [번호] 표기
[참고 문헌]:
[1] 날짜 | 제목
[2] 날짜 | 제목
(최대 5개, 이후 즉시 종료)

질문: {question}

[금융 데이터]
{formatted_context}"""

    print(f">> {len(context_docs)}개 문서를 바탕으로 답변 생성 중...")

    # 범위 쿼리(장기 비교)는 토큰 더 허용, 단일 날짜는 1200으로 제한
    gen_max_tokens = 1500 if is_range_query else 1200
    raw_answer = ask_kanana(prompt, max_tokens=gen_max_tokens, temp=0.1)

    # 데이터 없음 상황은 명시적 안내 메시지
    _filler_patterns = [
        r"^네[,.]?\s*(언제든|추가로|궁금한)",
        r"^언제든지\s*(말씀|질문)",
        r"^가능한\s*범위\s*내에서",
        r"^더\s*(궁금|알고\s*싶)",
    ]
    _is_filler = any(
        re.search(p, raw_answer.strip(), re.MULTILINE)
        for p in _filler_patterns
    )
    if _is_filler:
        answer = "요청하신 기간의 데이터가 제공된 자료에 없어 정확한 답변을 드리기 어렵습니다. 다른 날짜나 지표로 다시 질문해 주세요."
        print(">> [Skip] 마무리 멘트 응답 감지 → 안내 메시지로 교체.")
        return {"answer": answer}

    # 텍스트 클리닝 함수 적용
    answer = _clean_answer(raw_answer)
    print(">> 답변 생성 완료.")

    return {"answer": answer}



# ── Hallucination Grader ─────────────────────────────────────────────────
def hallucination_grader_node(state: AgentState):
    print("\n--- [NODE] Hallucination Grader ---")
    answer = state.get("answer")
    documents = state.get("retrieved_docs", [])
    topic = state.get("topic", "finance")
    retry_count = state.get("retry_count", 0)

    if topic != "finance" or not documents:
        print(">> [Pass] 일반 대화 또는 검증할 문서 없음.")
        return {
            "hallucination_score": "yes", 
            "analysis_note": "검증할 데이터 부족으로 패스",
        }

    context_text = ""
    for i, doc in enumerate(documents):
        d_date = doc.metadata.get('date', 'unknown')
        d_item = doc.metadata.get('item', '지표')
        context_text += f"문서 [{i+1}] ({d_date}, {d_item}): {doc.page_content}\n"

    # 웹 문서 비중 확인 (웹 문서가 절반 이상이면 기준 완화)
    web_doc_count = sum(1 for d in documents if d.metadata.get("item") == "웹데이터")
    is_mostly_web = web_doc_count >= len(documents) // 2

    # "데이터 없음" 스킵은 순수 DB 답변에만 적용
    _no_data_markers = ["찾을 수 없습니다", "데이터가 없", "정보가 없", "관련 정보를 찾"]
    answer_str = str(answer)
    if not is_mostly_web and any(m in answer_str for m in _no_data_markers):
        print(">> [Skip] 데이터 없음 응답 → 환각 검사 생략, PASS 처리")
        return {"hallucination_score": "yes", "analysis_note": "데이터 없음 응답 → PASS"}
    # 웹 기반 답변이 "데이터 없음"만으로 구성된 경우 스킵 허용
    if is_mostly_web and all(m in answer_str for m in ["찾을 수 없습니다"]) and len(answer_str.strip()) < 100:
        print(">> [Skip] 웹 기반 데이터 없음 단독 응답 → PASS 처리")
        return {"hallucination_score": "yes", "analysis_note": "데이터 없음 응답 → PASS"}

    if is_mostly_web:
        # 웹 검색 기반 답변은 구조 검증만 수행
        prompt = f"""아래 [답변]이 금융 분야에 관한 내용인지, 그리고 금융과 전혀 무관한 내용이 포함되어 있는지 확인하세요.

[답변]
{str(answer)[:600]}

판정 기준:
- 금융 관련 내용(환율, 금리, 주가, 시황 등)이 주된 내용이면: PASS
- 금융과 무관한 내용이 섞여 있으면: FAIL

첫 줄에 PASS 또는 FAIL 중 하나만 출력하세요."""
        res = ask_kanana(prompt, max_tokens=10, temp=0.0)
        first_token = "PASS" if "PASS" in res.strip().upper() else "FAIL"
    else:
        # DB 기반 답변은 수치 정합성 검증
        prompt = f"""아래 [답변]의 수치가 [근거 문서]에 있는 수치와 일치하는지 확인하세요.

[근거 문서] (핵심 수치만 확인)
{context_text[:1200]}

[답변]
{str(answer)[:600]}

판정 기준:
- 답변의 수치가 문서 수치와 같거나 문서에서 계산 가능하면: PASS
- 답변에 문서에 없는 수치가 명시되어 있으면: FAIL
- 답변에 수치가 전혀 없으면: PASS

첫 줄에 PASS 또는 FAIL 중 하나만 출력하세요."""
        res = ask_kanana(prompt, max_tokens=10, temp=0.0)
        first_token = "PASS" if "PASS" in res.strip().upper() else "FAIL"

    score = "yes" if first_token == "PASS" else "no"

    if score == "yes":
        print(">> [결과] 환각 없음 (Faithful)")
        return {
            "hallucination_score": "yes",
            "analysis_note": "검증 완료: 근거 문서의 팩트와 일치합니다.",
        }
    else:
        print(f">> [결과] 환각 발견! (현재 재시도 횟수: {retry_count}/2)")

        if retry_count >= 2:
            print(">> [Alert] 최대 교정 횟수를 초과하여 면책 문구를 덧붙입니다.")
            disclaimer = "\n\n---\n**[주의]** 위 답변 일부 수치는 원본 데이터와 차이가 있을 수 있습니다. 중요한 투자 결정 전에 원문을 직접 확인하세요."
            final_answer = str(answer) + disclaimer
            return {
                "hallucination_score": "no",
                "analysis_note": f"[필독 수정 지시]: {res}",
                "answer": final_answer,
            }

        return {
            "hallucination_score": "no",
            "analysis_note": f"[필독 수정 지시]: {res}",
        }


def route_hallucination(state: AgentState):
    score = state.get("hallucination_score", "no")
    retry = state.get("retry_count", 0)

    if score == "yes":
        print("--- [ROUTE] 검증 통과. 최종 답변 확정. ---")
        return "Faithful"
    if retry >= 2:
        print("--- [ROUTE] 최대 교정 횟수 도달. 강제 종료. ---")
        return "Faithful"
    print(f"--- [ROUTE] 환각 감지 (재시도: {retry}/2) → 교정 루프 ---")
    return "Hallucination_Detected"


# ── Answer Regenerator ──────────────────────────────────
def answer_regenerator_node(state: AgentState):
    print("\n--- [NODE] Answer Regenerator (Self-Correction Loop) ---")
    question = state["question"]
    context_docs = state.get("retrieved_docs", [])
    previous_answer = state.get("answer", "")
    feedback = state.get("analysis_note", "데이터 무결성 및 인용 번호 재검토 필요")
    target_date = state.get("target_date", "")
    current_retry = state.get("retry_count", 0)

    formatted_context = ""
    for i, doc in enumerate(context_docs):
        formatted_context += (
            f"문서 [{i+1}] (항목: {doc.metadata.get('item','지표')}, "
            f"날짜: {doc.metadata.get('date','unknown')})\n"
            f"내용: {doc.page_content}\n\n"
        )

    prompt = f"""당신은 금융 보고서 전문 교정 분석가입니다.
이전 답변에서 사실 왜곡, 수치 오류, 인용 오기가 발견되었습니다.
아래의 [핵심 교정 피드백]을 철저히 반영하여 보고서를 다시 작성하세요.

[금지사항]
1. "[수정 내역]", "[교정 피드백]", "[생성된 답변]" 등 교정 과정에 대한 설명이나 메타 정보를 절대로 출력하지 마세요.
2. 오직 사용자에게 바로 보여줄 [요약], [상세 분석], [참고 문헌] 섹션만 깔끔하게 출력하세요.
3. 근거 문서에 없는 외부 지식(예: 미국 금리 인상, 글로벌 경제 둔화 등)을 임의로 덧붙이지 마세요.

[핵심 교정 피드백]
{feedback}

[재작성 가이드라인]
1. 지적받은 수치를 [정확한 근거 문서]의 정확한 값으로 교정하세요.
2. 인용 번호([1], [2])가 실제 해당 정보가 포함된 문서 번호와 일치하는지 확인하세요.
3. [요약] / [상세 분석] / [참고 문헌] 구조를 반드시 유지하세요.
4. 마크다운 강조(**)를 사용하지 마세요. 평문으로 작성하세요.
5. 문서에 없는 내용을 추측하거나 지어내지 마세요.

오늘 날짜: {target_date}

사용자 질문: {question}

[이전 답변]
{previous_answer}

[정확한 근거 문서]
{formatted_context}"""

    print(f">> {current_retry + 1}차 재작성 시작...")
    
    raw_answer = ask_kanana(prompt, max_tokens=1500, temp=0.3)
    cleaned_answer = _clean_answer(raw_answer)
    
    print(">> 재작성 완료.")

    return {"answer": cleaned_answer, "retry_count": 1}


# ── 그래프 정의 ────────────────────────────────────────────────────────────────
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("input_router",          input_router_node)
    workflow.add_node("multi_query_generator", multi_query_generator_node)
    workflow.add_node("check_availability",    check_availability_node)
    workflow.add_node("rag_searcher",          rag_searcher_node)
    workflow.add_node("web_searcher",          web_searcher_node)
    workflow.add_node("context_filter",        context_filter_node)
    workflow.add_node("context_reranker",      context_reranker_node)
    workflow.add_node("context_evaluator",     context_evaluator_node)
    workflow.add_node("answer_generator",      answer_generator_node)
    workflow.add_node("hallucination_grader",  hallucination_grader_node)
    workflow.add_node("answer_regenerator",    answer_regenerator_node)

    workflow.set_entry_point("input_router")

    # [A] 입력 라우팅
    workflow.add_conditional_edges(
        "input_router", route_after_input,
        {"finance_search": "multi_query_generator", "direct_answer": "answer_generator"},
    )

    # [B] 분석 단계
    workflow.add_edge("multi_query_generator", "check_availability")
    workflow.add_edge("check_availability",    "rag_searcher")

    # [C] DB 결과에 따른 웹 검색 여부
    workflow.add_conditional_edges(
        "rag_searcher",
        lambda x: "Web_Required" if x.get("is_fallback") else "Internal_Only",
        {"Web_Required": "web_searcher", "Internal_Only": "context_filter"},
    )

    # [D] 문서 가공 파이프라인
    workflow.add_edge("web_searcher",   "context_filter")
    workflow.add_edge("context_filter", "context_reranker")
    workflow.add_edge("context_reranker", "context_evaluator")

    # [E] 정보 충분성 평가
    workflow.add_conditional_edges(
        "context_evaluator", route_from_evaluator,
        {"Enough": "answer_generator", "Not_Enough": "web_searcher"},
    )

    # [F] 환각 체크 루프 (answer_regenerator 독립 노드로 분리)
    workflow.add_edge("answer_generator", "hallucination_grader")
    workflow.add_conditional_edges(
        "hallucination_grader", route_hallucination,
        {"Faithful": END, "Hallucination_Detected": "answer_regenerator"},
    )
    workflow.add_edge("answer_regenerator", "hallucination_grader")

    return workflow.compile()


graph = build_graph()

# ── FastAPI 앱 초기화 ───────────────────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel

api = FastAPI(title="Kanana 금융 에이전트 API", version="0.1.0")

# ── 서버 시작 시 DB 및 메타데이터 초기화 ──────────────────────────────────────
@api.on_event("startup")
def startup():
    global vector_db
    vector_db = Chroma(persist_directory=Config.DB_PATH, embedding_function=get_embeddings())
    ensure_date_int_metadata()
    log.info("서버 시작 완료. DB 및 메타데이터 초기화 완료.")

# ── 요청/응답 모델 ─────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    answer: str
    elapsed: str

# ── 헬스체크 ───────────────────────────────────────────────────────────────────
@api.get("/health")
def health():
    return {"ok": True}

# ── 질문 엔드포인트 ────────────────────────────────────────────────────────────
@api.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question
    inputs = {
        "question": question,
        "loop_count": 0,
        "retry_count": 0,
        "target_date_int": 0,
        "start_date_int":  0,
        "end_date_int":    0,
    }

    start_time   = datetime.now()
    result       = graph.invoke(inputs, {"recursion_limit": 50})
    elapsed      = datetime.now() - start_time
    elapsed_str  = f"{int(elapsed.total_seconds() // 60)}분 {int(elapsed.total_seconds() % 60)}초"
    final_answer = result.get("answer", "(답변 없음)")

    log.info("=" * 60)
    log.info(f"[질문] {question}")
    log.info(f"[소요시간] {elapsed_str}")
    log.info(f"[답변]\n{final_answer}")
    log.info("=" * 60)

    return AskResponse(
        question=question,
        answer=final_answer,
        elapsed=elapsed_str,
    )

# ── 진입점 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:api", host="0.0.0.0", port=Config.APP_PORT, reload=False)
    