"""
Run this to test if Groq API is working correctly.
Usage: python test_groq.py
"""
from groq import Groq
from config import Config

print(f"API Key: {Config.GROQ_API_KEY[:12]}...")

try:
    client = Groq(api_key=Config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_tokens=50,
    )
    print("✅ Groq working! Response:", response.choices[0].message.content)
except Exception as e:
    print(f"❌ Error type: {type(e).__name__}")
    print(f"❌ Error detail: {e}")