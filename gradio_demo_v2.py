import gradio as gr
import numpy as np
from utils import ASRInference  # Replace 'your_module' with the actual module where ASRInference is defined

# Create an instance of ASRInference
asr = ASRInference()

# Define a function that uses the ASRInference instance

from faster_whisper import WhisperModel

model = WhisperModel("large-v2")

def transcribe_audio(audio):
  segments, info = model.transcribe(audio)
  for segment in segments:
    text = segment.text
  return text

def sentiment_analyse_text(audio):
  text = transcribe_audio(audio)
  from transformers import pipeline
  pipe = pipeline("text-classification", model="Peed911/french_sentiment_analysis")
  sentiment = pipe(text)
  final_text = "the model predict {} with {} as probability ".format(sentiment[0]["label"], sentiment[0]["score"])
  return final_text
# Create a Gradio interface
iface = gr.Interface(fn=transcribe_audio, inputs="audio", outputs="text")

if __name__ == "__main__":
    iface.launch(debug = True)
    #print(transcribe_audio("audio_test.wav"))