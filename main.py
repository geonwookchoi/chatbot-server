import os
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv # 👈 1. 이 줄 추가

load_dotenv() # 👈 2. 이 줄 추가

# --- 1. 설정 ---
# .env 파일 등에서 API 키를 안전하게 로드하는 것이 좋습니다.
# 여기서는 환경 변수를 사용합니다.
openai.api_key = os.environ.get("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

# --- 2. FastAPI 앱 초기화 ---
app = FastAPI()

# --- 3. 데이터 모델 정의 ---
# 안드로이드 앱에서 보낼 데이터의 형식을 Pydantic으로 정의합니다.
# 이렇게 하면 FastAPI가 자동으로 데이터 유효성을 검사해줍니다.
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] # [{"role": "user", "content": "안녕?"}, ...] 형태의 대화 기록

class ChatResponse(BaseModel):
    reply: str

# --- 4. API 엔드포인트(URL 경로) 생성 ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    안드로이드 앱으로부터 채팅 메시지와 대화 기록을 받아
    OpenAI API에 전달하고, 봇의 응답을 반환하는 엔드포인트
    """
    # 시스템 메시지: 챗봇의 정체성, 역할 등을 정의합니다.
    system_message = {"role": "system", "content": "당신은 사용자의 고민을 들어주고 공감해주는 상담 챗봇입니다. 사용자에게 친절하고 따뜻하게 응답해주세요."}
    
    # OpenAI에 보낼 전체 메시지 리스트 생성
    # [시스템 메시지, 이전 대화들, 새로운 사용자 메시지]
    messages_to_send = [system_message] + request.history + [{"role": "user", "content": request.message}]

    try:
        # OpenAI API 호출
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages_to_send,
            temperature=0.7,
            max_tokens=500
        )
        
        bot_reply = response.choices[0].message.content
        return ChatResponse(reply=bot_reply)

    except Exception as e:
        # API 호출 중 오류 발생 시 500 에러를 반환
        print(f"OpenAI API Error: {e}")
        raise HTTPException(status_code=500, detail="OpenAI API 처리 중 오류가 발생했습니다.")