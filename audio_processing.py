from openai import OpenAI
from config import API_KEY

def transcribe_audio(file_path):
    client = OpenAI(api_key=API_KEY)
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
          model="whisper-1", 
          file=audio_file
        )
    return transcription.text
