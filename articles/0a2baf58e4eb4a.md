---
title: "【実践】OpenAIからOpenRouterへの移行ガイド"
emoji: "💰"
type: "tech"
topics: [OpenRouter, OpenAI, LangChain, Python]
published: true
---

# OpenAIの料金に驚愕した話

こんにちは。QiitaからZennに移行してきたEijiです。

ある日、OpenAIのAPIを実行しようとしたら...

**「クレジットの期限が切れました」**

えっ、マジで！？調べてみたら、なんとクレジットには1年の制限があるとのこと。これは痛い。。。

そこで見つけた救世主が[OpenRouter](https://openrouter.ai/)です。

# なぜOpenRouterがいいのか

1. **モデル選択の自由**: GPT-4だけでなく、Claude 3やLlama 2も使える
2. **統一インターフェース**: 全部のAPIを同じ方法で呼べる
3. **課金管理の簡素化**: 複数のAI事業者への支払いが一本化

# 実装手順

## 1. 初期設定

初期設定は本当に簡単である。

1. pythonの適当なプロジェクトを作成し、下記をpipでインストール
    1. langchain_openai
    1. langchain_core.messages
1. OpenRouterに初期設定を行う。（サインアップ / 初期課金 / API KEYの作成）
1. OPENROUTER_API_KEYという名前で、上記で作成したAPI KEYを環境変数に設定

## 2. 共通メソッドを作成

これが一番のポイント。たった3行でOpenAIからの移行が完了します：

```python
from langchain_openai import ChatOpenAI
from os import getenv

def get_openrouter(model: str = "anthropic/claude-3-sonnet") -> ChatOpenAI:
    """OpenRouter APIを使用してLLMにアクセスするためのChatOpenAIインスタンスを返す

    OpenRouterは複数のLLMプロバイダーへの統一的なアクセスを提供するサービスです。
    直接OpenAIを使用する場合は、ChatOpenAI(model="gpt-4")のように指定することもできます。

    Args:
        model: 使用するモデル名。デフォルトはClaude 3 Sonnet

    Returns:
        ChatOpenAI: 設定済みのChatOpenAIインスタンス
    """
    return ChatOpenAI(
        model=model,
        openai_api_key=getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )

```

## 3. 実践的な使用例

理論はこれだけ。では実際にどう使うのか？面白い例を用意しました：

```python
import random
from langchain_core.messages import HumanMessage
from llm_util import get_openrouter

# 異なるモデルを試して、回答の違いを比較
models = [
    "anthropic/claude-3-sonnet",
    "google/gemini-pro",
    "meta-llama/llama-2-70b-chat",
    "nousresearch/deephermes-3-llama-3-8b-preview:free"
]

question = "日本のIT業界の給料を上げるには、どうすればいいですか？"

print("=== モデル別の回答比較 ===\n")

for model_name in models:
    try:
        model = get_openrouter(model=model_name)
        messages = [HumanMessage(content=question)]
        response = model.invoke(messages)
        print(f"🤖 {model_name}の回答:\n{response.content}\n")
    except Exception as e:
        print(f"❌ {model_name}でエラー: {str(e)}\n")
```

簡単に、呼び出すものが作れる。

# まとめ

- 複数モデルを使い分けることで、用途に応じた最適な選択が可能に
- LangChainとの相性も抜群
- 支払先が一つになるので、管理が楽になる
