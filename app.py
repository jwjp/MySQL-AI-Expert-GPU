# TODO:
#   0. .\venv\Scripts\activate 가상 환경 활성화
#   1. python ingest.py 실행하여 지식 데이터베이스(db 폴더) 생성 (최초 1회)
#   2. python finetune.py 실행하여 맞춤형 AI 어댑터(mysql-expert-llama-3-8b 폴더) 생성 (최초 1회)
#   3. uvicorn main:app --reload --host 127.0.0.1 --port 8000 실행하여 백엔드 API 서버 시작
#   4. streamlit run app.py 실행하여 프론트엔드 웹 UI 시작

import streamlit as st
import requests

# FastAPI 서버 주소
API_URL = "http://127.0.0.1:8000/ask"

st.title("MySQL 8.0 전문가 AI 챗봇 🤖")
st.markdown("---")

# 채팅 기록을 세션 상태에 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("MySQL 8.0에 대해 무엇이든 물어보세요!"):
    # 사용자 메시지를 채팅 기록에 추가하고 화면에 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답을 위한 준비
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("답변을 생각하는 중... 🤔")
        
        try:
            # API 서버에 질문 전송
            response = requests.post(API_URL, json={"query": prompt})
            response.raise_for_status() # 오류 발생 시 예외 처리
            
            ai_answer = response.json()["answer"]
            message_placeholder.markdown(ai_answer)
            
            # AI 응답을 채팅 기록에 추가
            st.session_state.messages.append({"role": "assistant", "content": ai_answer})
            
        except requests.exceptions.RequestException as e:
            error_message = f"오류가 발생했습니다: {e}"
            message_placeholder.markdown(error_message)