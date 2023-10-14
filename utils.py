from transformers import AutoProcessor, AutoModelForCTC
import torch
import numpy as np
#import librosa

class ASRInference:
    def __init__(self, model_name="openai/whisper-large-v2"):
        self.model = AutoProcessor.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def inference(self, audio):
        _, audio_data = audio

        audio_tensor = torch.Tensor(audio_data)  # Convert audio to a PyTorch tensor
        #print("audio tensor ", audio_tensor)
        inputs =  self.processor(audio_tensor, sampling_rate=16_000, return_tensors="pt")
        #print("input ",inputs)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=2)[0]  # Changed dim to 2 for CTC output
        text = self.processor.decode(predicted_ids, skip_special_tokens=True).lower()
        return text
    def inference_faspi(self, audio):

        audio_tensor = torch.Tensor(audio)  # Convert audio to a PyTorch tensor
        #print("audio tensor ", audio_tensor)
        inputs =  self.processor(audio_tensor, sampling_rate=16_000, return_tensors="pt")
        #print("input ",inputs)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=2)[0]  # Changed dim to 2 for CTC output
        text = self.processor.decode(predicted_ids, skip_special_tokens=True).lower()
        return text