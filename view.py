
#from pyexpat.errors import messages
from httpx import stream
import streamlit as st
from openai import OpenAI #openaiのSDK

st.set_page_config(page_title = "Local LLM Chat")


st.sidebar.title("設定")
model = st.sidebar.text_input("モデル名", value="llama3.2:3B")
temperature = st.sidebar.slider("temperature",0.0, 2.0, 0.3, 0.1)

system_prompt = st.sidebar.text_area(
    "System Prompt",
    "あなたは有能なアシスタントです。日本語で回答してください。"
)


#タイトル
st.title("Local LLM Chat")
st.text(system_prompt)


#会話の履歴を保管
if "messages" not in st.session_state:
    st.session_state.messages = []

#会話の履歴をリセットするボタン
if st.sidebar.button("会話をリセット"):
    st.session_state.messages = []

#会話の履歴を表示
for m in  st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])


prompt = st.chat_input("メッセージを入力")

# OpenAI SDK（クライアント）を使って、Ollama のローカル API にアクセスしている
# ✔ なぜ OpenAI SDK を使って Ollama に接続できるのか
# Ollama は OpenAI API と同じ形式のエンドポイントを提供しているから。
# 例：
# /v1/chat/completions
# /v1/models
# だから、OpenAI SDK をそのまま使える
# - from openai import OpenAI → OpenAI SDK の読み込み
# - base_url を変える → 接続先を Ollama に切り替える
# - 実際に動いているのは ローカルの llama3.2:3b（Ollama）
client = OpenAI(
    api_key = "ollama",
    base_url = "http://localhost:12000/v1"
)

#初期状態のエラー回避のためif文追加
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    #ユーザのプロンプトを表示
    with st.chat_message("user"):
        st.write(prompt)

    if system_prompt.strip():
        #システムプロンプトも含めてollamaに渡す
        messages = [{"role" : "user","content": system_prompt}] + st.session_state.messages
    else:
        #システムプロンプトの指示が無いときはそのままプロンプトだけを使う
        messages = st.session_state.messages 


    #LLMの返答を表示
    with st.chat_message("assistant"):
        placeholder = st.empty()
        stream_response = ""
        stream = client.chat.completions.create(
        model = model,
        messages = messages,
        temperature = temperature,
        stream = True
    )
        
    
    for chunk in stream:
        stream_response += chunk.choices[0].delta.content
        placeholder.write(stream_response)

    #会話の履歴を保存
    st.session_state.messages.append({"role": "assistant", "content": stream_response})



