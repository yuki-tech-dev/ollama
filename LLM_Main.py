from openai import OpenAI

client = OpenAI(
    api_key = "ollama",
    base_url="http://localhost:12000/v1"
)

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role":"user","content":"こんにちは、トヨタ自動車について教えて日本語で回答して"}],
    temperature=0.1
)

#レスポンスには色々な情報が入ってくる。
#インデックス0番目の.messages.contentだけを取り出すと、テキストのみを取り出すことが可能である。
print("response全体：",response)
print("テキストだけ抽出：",response.choices[0].message.content)




