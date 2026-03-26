"""
Run: python test_tts.py
Opens test.mp3 if successful
"""
import asyncio
import edge_tts

async def test():
    text  = "வணக்கம், நான் பிக்ஸி, உங்கள் உதவியாளர்"
    voice = "ta-IN-PallaviNeural"
    print(f"Testing voice: {voice}")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("test_tamil.mp3")
    print("✅ Saved to test_tamil.mp3 — play this file to check Tamil audio")

asyncio.run(test())