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
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Upload and convert audio file if necessary
from google.colab import files
uploaded = files.upload()
path = next(iter(uploaded))

# Parameters
num_speakers = 2  # Number of speakers
language = 'English'  # Language for transcription
model_size = 'large'  # Model size for Whisper

# Set model name
model_name = model_size
if language == 'English' and model_size != 'large':
    model_name += '.en'

# Load Whisper model
model = whisper.load_model(model_size)

# Convert non-wav files to wav if necessary
if path[-3:] != 'wav':
    subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
    path = 'audio.wav'

# Get the duration of the audio file
with contextlib.closing(wave.open(path, 'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

# Initialize pyannote speaker verification model
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb", device=torch.device("cuda"))

# Initialize pyannote audio utility
audio = Audio()

# Transcribe the audio using Whisper
result = model.transcribe(path)
segments = result["segments"]

# Extract speaker embeddings for each segment
def segment_embedding(segment):
    start = segment["start"]
    end = min(duration, segment["end"])  # Handle overshooting end timestamp
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    if waveform.shape[0] > 1:  # If the waveform has more than one channel, average them to mono
        waveform = waveform.mean(axis=0, keepdims=True)
    return embedding_model(waveform[None])

# Generate embeddings for each segment
embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
    embeddings[i] = segment_embedding(segment)

# Handle any NaN values in embeddings
embeddings = np.nan_to_num(embeddings)

# Perform Agglomerative Clustering for speaker diarization
clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_

# Assign speaker labels to segments
for i in range(len(segments)):
    segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

# Helper function to format time in HH:MM:SS
def time(secs):
    return str(datetime.timedelta(seconds=round(secs)))

# Write transcript with speaker labels to a file
with open("transcript.txt", "w") as f:
    for i, segment in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            f.write(f"\n{segment['speaker']} {time(segment['start'])}\n")
        f.write(segment["text"][1:] + ' ')
