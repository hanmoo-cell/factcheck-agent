import os
import time
import requests
import streamlit as st
from ddgs import DDGS
from google import genai
from google.genai import types

# ==========================================
# 1. 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="근거 검증형 자료조사 에이전트 AI",
    page_icon="🔍",
    layout="wide",
)
st.title("🔍 근거 검증형 자료조사 에이전트 AI")
st.caption("생성형 AI 환각 완화 후속 탐구 | 웹 검색(ddgs) + HTTP 링크 생존 검증 + 수동 ReAct 루프")


# ==========================================
# 2. API 키 안전 로드 (Streamlit Secrets / 환경변수)
# ==========================================
def get_default_api_key() -> str:
    try:
        return st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    except Exception:
        return os.environ.get("GEMINI_API_KEY", "")


# ==========================================
# 3. 사이드바: 실험 비교 변수 설정
# ==========================================
with st.sidebar:
    st.header("⚙️ 실험 변수 설정")
    api_key = st.text_input(
        "Gemini API Key",
        value=get_default_api_key(),
        type="password",
        help="Streamlit Cloud 배포 시 Secrets에 GEMINI_API_KEY를 등록하면 자동 입력됩니다.",
    )

    mode = st.radio(
        "실행 모드 선택",
        ["에이전트 모드 (도구+검증)", "기본 LLM 모드 (도구 없음)"],
        help="동일한 질문을 두 모드로 실행해 환각 발생 여부와 응답 시간을 비교합니다.",
    )

    model_name = st.selectbox(
        "모델 선택",
        ["gemini-3.5-flash-lite", "gemini-3.8-flash"],
        help="무료 티어에서는 flash-lite 모델의 일일 호출 한도가 가장 넉넉합니다.",
    )

    temperature = st.slider(
        "Temperature (출력 무작위성)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0.0에 가까울수록 사실 중심의 일관된 답변을 생성합니다.",
    )

    max_steps = st.slider(
        "최대 추론 반복 횟수 (무한 루프 방지)",
        min_value=1,
        max_value=6,
        value=4,
    )

    st.divider()
    if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ==========================================
# 4. 에이전트 도구(Tool) 2종 구현
# ==========================================
def search_web(query: str) -> dict:
    """웹에서 최신 정보나 문헌·사실을 검색하여 제목, URL, 요약 목록을 반환합니다."""
    try:
        raw_results = list(DDGS().text(query, max_results=4))
        if not raw_results:
            return {"status": "empty", "query": query, "results": []}

        cleaned = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw_results
        ]
        return {"status": "ok", "query": query, "results": cleaned}
    except Exception as e:
        return {"status": "error", "query": query, "message": str(e)}


def verify_url(url: str) -> dict:
    """URL에 실제 HTTP 요청을 보내 접속 가능 여부(200~399)와 상태 코드를 반환합니다."""
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return {"url": url, "alive": False, "status_code": None, "error": "유효하지 않은 URL 형식"}

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.head(url, headers=headers, allow_redirects=True, timeout=5)
        # 일부 서버는 HEAD 요청을 차단(403/405 등)하므로 GET으로 1회 재확인
        if resp.status_code >= 400:
            resp = requests.get(url, headers=headers, allow_redirects=True, timeout=5, stream=True)
            resp.close()

        is_alive = 200 <= resp.status_code < 400
        return {
            "url": url,
            "alive": is_alive,
            "status_code": resp.status_code,
        }
    except Exception as e:
        return {
            "url": url,
            "alive": False,
            "status_code": None,
            "error": str(e),
        }


TOOL_MAP = {
    "search_web": search_web,
    "verify_url": verify_url,
}

SYSTEM_PROMPT = """당신은 수행평가와 학술 자료조사를 돕는 '근거 검증형 에이전트 AI'입니다.
반드시 아래 규칙을 지키세요:
1. 사실 확인이 필요한 질문은 추측하지 말고 반드시 `search_web` 도구를 호출해 근거를 찾으세요.
2. 답변에 URL 출처를 포함할 때는 반드시 `verify_url` 도구로 `alive: true`가 확인된 링크만 제시하세요.
3. 검색 결과가 없거나 검증된 근거가 부족하면 절대 지어내지 말고, '확인할 수 있는 신뢰할 만한 근거를 찾지 못했습니다(불확실 인정).'라고 명확히 답하세요.
4. 최종 답변은 [핵심 답변], [검증된 출처], [신뢰도 판정(검증됨 / 불확실 인정)] 형식으로 정리하세요."""


# ==========================================
# 5. 대화 기록 렌더링
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])


# ==========================================
# 6. 사용자 입력 및 수동 ReAct 에이전트 루프 실행
# ==========================================
prompt = st.chat_input("질문을 입력하세요 (예: 명량해전 거북선 12척 전술은? / 궁중 요리책 《수라간비결》 조리법은?)")

if prompt:
    if not api_key:
        st.error("사이드바에 Gemini API Key를 입력하거나 Streamlit Secrets에 등록해 주세요.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = genai.Client(api_key=api_key)
    start_time = time.time()
    tool_call_count = 0
    final_text = ""

    with st.chat_message("assistant"):
        # [모드 A] 기본 LLM 모드 (도구·검증 없음)
        if mode.startswith("기본 LLM"):
            with st.spinner("기본 LLM 응답 생성 중..."):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=temperature),
                    )
                    final_text = response.text or "(모델이 빈 응답을 반환했습니다.)"
                except Exception as e:
                    final_text = f"⚠️ API 호출 오류가 발생했습니다:\n```text\n{e}\n```"

        # [모드 B] 수동 ReAct 에이전트 모드 (도구 호출 + 링크 검증 루프)
        else:
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
            ]
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=temperature,
                tools=[search_web, verify_url],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            final_text = "최대 추론 반복 횟수에 도달하여 답변을 확정하지 못했습니다(불확실 인정)."

            with st.status("🤖 에이전트 추론 및 근거 검증 진행 중...", expanded=True) as status:
                for step in range(1, max_steps + 1):
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            config=config,
                        )
                    except Exception as e:
                        status.update(label="⚠️ API 호출 오류 발생", state="error")
                        final_text = f"⚠️ API 호출 오류가 발생했습니다:\n```text\n{e}\n```"
                        break

                    if not response.candidates or not response.candidates[0].content:
                        status.update(label="⚠️ 모델 응답 없음", state="error")
                        final_text = "모델이 유효한 응답을 반환하지 못했습니다(불확실 인정)."
                        break

                    candidate_content = response.candidates[0].content
                    contents.append(candidate_content)

                    # 도구 호출 요청이 없으면 최종 답변으로 확정하고 루프 종료
                    if not response.function_calls:
                        final_text = response.text or final_text
                        status.update(
                            label=f"✅ 검증 완료 (총 {step}단계, 도구 호출 {tool_call_count}회)",
                            state="complete",
                        )
                        break

                    # 요청된 도구(search_web, verify_url) 순차 실행 및 실시간 시각화
                    tool_response_parts = []
                    for fc in response.function_calls:
                        tool_call_count += 1
                        fn_name = fc.name
                        fn_args = dict(fc.args) if fc.args else {}

                        st.write(f"**[Step {step}] 도구 호출:** `{fn_name}({fn_args})`")
                        if fn_name in TOOL_MAP:
                            tool_result = TOOL_MAP[fn_name](**fn_args)
                        else:
                            tool_result = {"status": "error", "message": f"알 수 없는 도구: {fn_name}"}

                        st.json(tool_result, expanded=False)

                        tool_response_parts.append(
                            types.Part.from_function_response(
                                name=fn_name,
                                response={"result": tool_result},
                            )
                        )

                    # Gemini API 규격에 맞게 role="user"로 도구 실행 결과 전달
                    contents.append(types.Content(role="user", parts=tool_response_parts))
                else:
                    status.update(
                        label=f"⚠️ 최대 반복 횟수({max_steps}회) 도달로 종료",
                        state="error",
                    )

        elapsed = time.time() - start_time
        meta_text = (
            f"⏱️ 응답 시간: {elapsed:.2f}초 | 🎛️ 실행 모드: {mode} | "
            f"🌡️ Temperature: {temperature:.1f} | 🛠️ 도구 호출: {tool_call_count}회"
        )

        st.markdown(final_text)
        st.caption(meta_text)
        st.session_state.messages.append(
            {"role": "assistant", "content": final_text, "meta": meta_text}
        )
