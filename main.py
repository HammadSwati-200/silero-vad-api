from fastapi import FastAPI, UploadFile, HTTPException
import torch
import soundfile as sf
import io
import numpy as np

app = FastAPI()

# Load Silero VAD model (torch.hub)
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)

# Silero utils now returns 5 items (adjusted)
(get_speech_timestamps, _, read_audio, _, _) = utils

@app.get("/")
def root():
    return {"status": "silero-vad-api", "ready": True}

@app.post("/analyze")
async def analyze(file: UploadFile):
    try:
        data = await file.read()
        # Read audio bytes via soundfile (returns numpy array and sample rate)
        wav, sr = sf.read(io.BytesIO(data), dtype='float32')
        # If stereo, convert to mono
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        # Convert numpy -> torch tensor
        wav_tensor = torch.from_numpy(wav).float()

        # Get speech timestamps (Silero returns sample indices)
        speech_timestamps = get_speech_timestamps(wav_tensor, model, sampling_rate=sr)

        # Convert segments to seconds for easier downstream usage
        segments_sec = [
            {"start_sec": float(seg["start"]) / sr, "end_sec": float(seg["end"]) / sr}
            for seg in speech_timestamps
        ]
        total_speech_sec = sum((seg["end"] - seg["start"]) for seg in speech_timestamps) / sr

        return {
            "speech_segments": segments_sec,
            "total_speech_duration_sec": float(total_speech_sec)
        }

    except Exception as e:
        # Return helpful error for debugging
        raise HTTPException(status_code=500, detail=str(e))
