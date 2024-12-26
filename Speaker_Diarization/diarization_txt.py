# prompt: can you write it so that i can save file in json and not .txt

from google.colab import files
import whisper
import datetime
import subprocess
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import json

# Install necessary libraries

# upload audio file
uploaded = files.upload()
path = next(iter(uploaded))
num_speakers = 2 #@param {type:"integer"}

language = 'English' #@param ['any', 'English']

model_size = 'large' #@param ['tiny', 'base', 'small', 'medium', 'large']


model_name = model_size
if language == 'English' and model_size != 'large':
  model_name += '.en'


embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))



if path[-3:] != 'wav':
  subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
  path = 'audio.wav'
model = whisper.load_model(model_size)
result = model.transcribe(path)
segments = result["segments"]
with contextlib.closing(wave.open(path,'r')) as f:
  frames = f.getnframes()
  rate = f.getframerate()
  duration = frames / float(rate)
audio = Audio()

def segment_embedding(segment):
  start = segment["start"]
  # Whisper overshoots the end timestamp in the last segment
  end = min(duration, segment["end"])
  clip = Segment(start, end)
  waveform, sample_rate = audio.crop(path, clip)
  if waveform.shape[0] > 1:  # Check if waveform has more than 1 channel
       waveform = waveform.mean(axis=0, keepdims=True) # Average channels to get mono
  return embedding_model(waveform[None])
embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
  embeddings[i] = segment_embedding(segment)

embeddings = np.nan_to_num(embeddings)
clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_
for i in range(len(segments)):
  segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
def time(secs):
  return datetime.timedelta(seconds=round(secs))

transcript = []
for (i, segment) in enumerate(segments):
  if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
    transcript.append({"speaker": segment["speaker"], "start": str(time(segment["start"])), "text": ""})
  transcript[-1]["text"] += segment["text"][1:] + ' '

# Save the transcript to a JSON file
with open('transcript.json', 'w') as f:
    json.dump(transcript, f, indent=4)

# Download the JSON file
files.download('transcript.json')