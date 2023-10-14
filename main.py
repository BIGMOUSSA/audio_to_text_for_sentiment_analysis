from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from faster_whisper import WhisperModel
from pydub import AudioSegment
from io import BytesIO
import soundfile as sf
# Load the tokenizer and model
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Peed911/french_sentiment_analysis")
model = AutoModelForSequenceClassification.from_pretrained("Peed911/french_sentiment_analysis")
label_mapping = {
    "Negative": 0,
    "Positive": 1
}
labels_name = [ "Negative", "Positive"]


model = WhisperModel("large-v2")


# Define the prediction function
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_names = labels_name[predicted_class]
    return predicted_class,  predicted_names

def transcribe_audio(audio):
  segments, info = model.transcribe(audio)
  for segment in segments:
    text = segment.text
  return text

app = FastAPI()

class Item(BaseModel):
    text : str 



@app.post("/predict")
def inference(file : UploadFile = File(...) ):
    audio, _ = sf.read(file.file)
    text = transcribe_audio(audio)
    return {"text" : text}

if __name__ == "__main__":
   audio_path = "common_voice_fr_17299384.wav"
   print(transcribe_audio(audio_path))