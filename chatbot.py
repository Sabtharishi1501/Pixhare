# ─────────────────────────────────────────────
# chatbot.py
# RAG pipeline — keyword search + Groq (no torch needed)
# ─────────────────────────────────────────────

import os, re
from groq import Groq
from knowledge_base import DOCUMENTS

# ── Config ──────────────────────────────────
try:
    from config import Config
    GROQ_API_KEY = Config.GROQ_API_KEY
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

MODEL_ID = "llama-3.3-70b-versatile"
TOP_K    = 3

_client  = None

def _get_groq_client():
    global _client
    if _client is None:
        if not GROQ_API_KEY or GROQ_API_KEY == "gsk_your_key_here":
            raise ValueError("GROQ_API_KEY not set in config.py")
        print(f"[RAG] Using Groq key: {GROQ_API_KEY[:8]}***")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _retrieve(query: str) -> list:
    """Keyword overlap scoring — no ML, no torch needed."""
    query_words = set(re.findall(r'\w+', query.lower()))
    scored = []
    for doc in DOCUMENTS:
        doc_words  = set(re.findall(r'\w+', doc['text'].lower()))
        overlap    = len(query_words & doc_words)
        id_boost   = len(query_words & set(doc['id'].replace('_',' ').split())) * 3
        scored.append((overlap + id_boost, doc['text'].strip()))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:TOP_K]]


def get_answer(query: str, chat_history: list = None, language: str = 'English') -> str:
    try:
        docs    = _retrieve(query)
        context = "\n\n".join(docs)

        lang_instruction = (
            f"CRITICAL: Reply ENTIRELY in {language}. "
            f"Every word must be in {language} script. "
            f"Technical terms like QR code, dashboard, email, upload — write phonetically in {language}. "
            f"Write in flowing sentences, no numbered lists, no English letters."
        ) if language != 'English' else (
            "Write in clear flowing sentences."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Pixi, a friendly helpful AI assistant for Pixhare — "
                    "an AI-powered event photo sharing platform. "
                    "Answer ONLY using the context provided. Be concise and friendly. "
                    f"Regardless of the language of the question, always reply in {language}. "
                    f"{lang_instruction}\n\n"
                    f"CONTEXT:\n{context}"
                )
            },
            {"role": "user", "content": query}
        ]

        client   = _get_groq_client()
        response = client.chat.completions.create(
            model       = MODEL_ID,
            messages    = messages,
            max_tokens  = 250,
            temperature = 0.2,
        )
        answer = response.choices[0].message.content.strip()
        return answer if answer else "I'm sorry, I couldn't generate a response."

    except ValueError as e:
        print(f"[RAG] Config error: {e}")
        return f"⚠️ {str(e)}"
    except Exception as e:
        print(f"[RAG] Error: {type(e).__name__}: {e}")
        return "I'm having trouble connecting. Please try again."


def initialize():
    """Lightweight init — just verify Groq connection."""
    try:
        _get_groq_client()
        print("[RAG] Chatbot initialized and ready.")
    except Exception as e:
        print(f"[RAG] Init warning: {e}")