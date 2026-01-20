#from pyexpat.errors import messages
from httpx import stream
import streamlit as st
from openai import OpenAI #openaiのSDK
import chromadb
from docx import Document
import requests

# chromadbの設定
DB_DIR = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_DIR)

if "collection" not in st.session_state:
    #chroma_clientを用いてテーブルを作るget or createなのであったらとる。なかったら作る
    st.session_state.collection = chroma_client.get_or_create_collection(
        name="local_docs",
        metadata={"hnsw:space": "cosine"} #緒方追加
    )



# ollamaからインストールしたモデルを使ったベクトル化関数
def ollama_embed(text):
    r = requests.post(
        "http://localhost:12000/api/embeddings",
        json={"model":"nomic-embed-text","prompt":text}
    )
    data = r.json()
    print(data["embedding"])
    return data["embedding"]


# Wordファイルを読み込む関数(テキストに抽出している)
def load_word_document(file):
    return "\n".join(para.text for para in Document(file).paragraphs)

# テキスト分割関数(チャンク化)
def split_text(text):
    chunk_size = 600
    overlap = 120
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks




# サイドバーの設定
st.set_page_config(page_title = "Local LLM Chat")

st.sidebar.title("設定")
model = st.sidebar.text_input("モデル名", value="llama3.2:3B")
temperature = st.sidebar.slider("temperature",0.0, 2.0, 0.3, 0.1)

system_prompt = st.sidebar.text_area(
    "System Prompt",
    "あなたは有能なアシスタントです。日本語で回答してください。"
)

# ワードファイルのアップロード
upload_files = st.sidebar.file_uploader("Wordファイルをアップロード(.docx)",
    type=["docx"],
    accept_multiple_files=True
)
    
if st.sidebar.button("インデックス作成"):
    for file in upload_files:
        text = load_word_document(file)
        chunks = split_text(text)
        for i, chunk in enumerate(chunks):
            embed = ollama_embed(chunk)
            st.session_state.collection.add(
                documents=[chunk],
                embeddings=[embed],
                ids=[f"{file.name}_{i}"]
            )
    st.sidebar.success("インデックス作成完了")


# タイトル
st.title("Local LLM Chat")
st.text(system_prompt)


# 会話の履歴を保管
if "messages" not in st.session_state:
    st.session_state.messages = []

# 会話の履歴をリセットするボタン
if st.sidebar.button("会話をリセット"):
    st.session_state.messages = []

# 会話の履歴を表示
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



print(prompt)
print("ガッツ")


# 初期状態のエラー回避のためif文追加してプロンプトがある場合のみ実行する
if prompt:
    #st.session_state.messages.append({"role": "user", "content": prompt})

    # ユーザのプロンプトを表示
    with st.chat_message("user"):
        st.write(prompt)

    
    # RAG検索
    query_embed = ollama_embed(prompt)
    results = st.session_state.collection.query(
        query_embeddings=[query_embed],
        n_results=2
    )

    filtered = [
    (doc, dist)
    for doc, dist in zip(results["documents"][0], results["distances"][0])
    if dist < 0.3
    ]


    print("distancesの中身:",results["distances"])

    print(results["documents"])
    print(results["documents"][0])
    print(filtered)




    # {
    #     "ids":
    #     "documents": [["doc1","doc2"]]
    #     "distances": [[xxx,xxx]]
    # ↓下の関連ドキュメントのところが、関連していない内容も引っ張ってきているみたいなので、
    # 改善方法を調べること。
    # }
    # if results["documents"][0]:# 最初はresults["documents"]であったが修正。
    if filtered:# 最初はresults["documents"]であったが修正。
        #context_text = "\n".join(results["documents"][0])
        context_text = "\n".join([doc for doc, dist in filtered])
        rag_prompt = f"""
        以下は関連ドキュメントの抜粋です。
        {context_text}
        この情報を参考に以下の質問に答えて下さい。
        以下の質問を、検索に最適な短いキーワードに変換してください。
        {prompt}
        """
        final_user_prompt = rag_prompt
    else:
        final_user_prompt = prompt

    st.session_state.messages.append({"role": "user", "content": final_user_prompt})


    if system_prompt.strip():
        # システムプロンプトも含めてollamaに渡す
        messages = [{"role" : "system","content": system_prompt}] + st.session_state.messages
    else:
        # システムプロンプトの指示が無いときはそのままプロンプトだけを使う
        messages = st.session_state.messages 


    # LLMの返答を表示
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

    # 会話の履歴を保存
    st.session_state.messages.append({"role": "assistant", "content": stream_response})
