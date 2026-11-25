import os
import time
import requests
import streamlit as st
from ddgs import DDGS
from google import genai
from google.genai import types

# 1. 페이지 기본 설정
st.set_page_config(page_title="근거 검증형 자료조사 에이전트 AI", page_icon="🔍", layout="wide")
st.title("🔍 근거 검증형 자료조사 에이전트 AI")
st.caption("생성형 AI 환각 완화 후속 탐구 | 웹 검색 + HTTP 링크 생존 검증 + 불확실성 인정 파이프라인")

# 2. 사이드바 설정 (실험 변수 제어)
with st.sidebar:
    st.header("⚙️ 실험 변수 설정")
    # 배포 시 Streamlit Secrets에서 자동 로드, 로컬 실행 시 직접 입력 가능
    default_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    api_key = st.text_input("Gemini API Key", value=default_key, type="password")

    mode = st.radio(
        "실행 모드 비교",
        ["근거 검증 에이전트 AI (도구+검증)", "기본 LLM (프롬프트·도구 없음)"],
        help="같은 질문을 두 모드로 실행해 환각 발생 여부를 비교할 수 있습니다."
    )
    model_name = st.selectbox(
        "모델 선택",
        ["gemini-3.5-flash-lite", "gemini-3.8-flash"],
        help="flash-lite는 무료 일일 한도(500회)가 넉넉합니다."
    )
    temperature = st.slider(
        "Temperature (출력 무작위성)",
        min_value=0.0, max_value=1.0, value=0.0, step=0.1,
        help="0.0에 가까울수록 사실 중심·일관된 답변을 생성합니다."
    )
    max_steps = st.slider("에이전트 최대 추론 반복 횟수", 1, 6, 4)

# 3. 에이전트가 사용할 도구(Tool) 함수 정의
def search_web(query: str) -> dict:
    """웹에서 최신 정보나 문헌·사실을 검색하여 제목, 요약, URL 목록을 반환합니다."""
    try:
        results = DDGS().text(query, max_results=4)
        if not results:
            return {"status": "empty", "results": []}
        cleaned = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in results
        ]
        return {"status": "ok", "results": cleaned}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def verify_url(url: str) -> dict:
    """URL에 실제 HTTP 요청을 보내 삭제되거나 가짜 링크가 아닌지(HTTP 200 여부) 검증합니다."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.head(url, headers=headers, allow_redirects=True, timeout=5)
        if resp.status_code >= 400:
            resp = requests.get(url, headers=headers, allow_redirects=True, timeout=5)
        return {
            "url": url,
            "alive": 200 <= resp.status_code < 400,
            "status_code": resp.status_code
        }
    except Exception as e:
        return {"url": url, "alive": False, "error": str(e)}

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

# 4. 대화 기록 초기화 및 출력
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. 질문 입력 및 에이전트 루프 실행
prompt = st.chat_input("예: 명량해전 거북선 12척 전술은? / 궁중 요리책 《수라간비결》 조리법 알려줘")

if prompt:
    if not api_key:
        st.error("사이드바에 Gemini API Key를 입력하거나 Secrets에 등록해 주세요.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = genai.Client(api_key=api_key)
    start_time = time.time()
    tool_call_count = 0

    with st.chat_message("assistant"):
        if mode.startswith("기본 LLM"):
            with st.spinner("기본 LLM 응답 생성 중..."):
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=temperature)
                )
                final_text = response.text or "(응답 없음)"
        else:
            # 수동 에이전트 루프 (ReAct 패턴 직접 구현)
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=temperature,
                tools=[search_web, verify_url],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            final_text = "최대 추론 횟수에 도달하여 답변을 확정하지 못했습니다(불확실 인정)."

            with st.status("🤖 에이전트 추론 및 근거 검증 진행 중...", expanded=True) as status:
                for step in range(1, max_steps + 1):
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    candidate_content = response.candidates[0].content
                    contents.append(candidate_content)

                    if not response.function_calls:
                        final_text = response.text or final_text
                        status.update(label=f"✅ 검증 완료 (총 {step}단계, 도구 호출 {tool_call_count}회)", state="complete")
                        break

                    tool_response_parts = []
                    for fc in response.function_calls:
                        tool_call_count += 1
                        fn_name = fc.name
                        fn_args = dict(fc.args) if fc.args else {}
                        st.write(f"**[Step {step}] 도구 호출:** `{fn_name}({fn_args})`")

                        result = TOOL_MAP[fn_name](**fn_args) if fn_name in TOOL_MAP else {"error": "Unknown tool"}
                        st.json(result, expanded=False)

                        tool_response_parts.append(
                            types.Part.from_function_response(
                                name=fn_name,
                                response={"result": result},
                            )
                        )
                    contents.append(types.Content(role="tool", parts=tool_response_parts))
                else:
                    status.update(label="⚠️ 최대 반복 횟수 도달 (무한 루프 방지 종료)", state="error")

        elapsed = time.time() - start_time
        st.markdown(final_text)
        st.caption(f"⏱️ 응답 시간: {elapsed:.2f}초 | 🎛️ 모드: {mode} | 🌡️ Temperature: {temperature} | 🛠️ 도구 호출: {tool_call_count}회")
        st.session_state.messages.append({"role": "assistant", "content": final_text})
