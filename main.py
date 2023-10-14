#from fastapi import FastAPI, File, UploadFile
#from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from faster_whisper import WhisperModel
#from pydub import AudioSegment
#from io import BytesIO
#import soundfile as sf
# Load the tokenizer and model
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from faster_whisper import WhisperModel

model = WhisperModel("large-v2")

# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("text-classification", model="Peed911/french_sentiment_analysis")

def transcribe_audio(audio):
  segments, info = model.transcribe(audio)
  for segment in segments:
    text = segment.text
  return text

def sentiment_analyse_audio(audio):
  """
  input : audio files

  output : text

  """
  text = transcribe_audio(audio)
  sentiment = pipe(text)
  final_text = "En analysant l'audio d'entrée, le modèle Whisper d'openai transcrit le texte suivant : \n << %s >>, \n et prédit un sentiment << %s >> avec << %.2f >> pourcent de probabilité "%(text, sentiment[0]["label"], 100*sentiment[0]["score"])
  return print(final_text)


if __name__ == "__main__":
   audio_path = "common_voice_fr_17299384.wav"
   print(sentiment_analyse_audio(audio_path))