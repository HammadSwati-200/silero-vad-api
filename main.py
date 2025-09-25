from fastapi import FastAPI, UploadFile
import torch
import soundfile as sf
import io

app = FastAPI()

# Load Silero VAD model
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)

(get_speech_timestamps, _, read_audio, _, _, _) = utils

@app.post("/analyze")
async def analyze(file: UploadFile):
    # Read audio file
    audio_bytes = await file.read()
    wav, sr = sf.read(io.BytesIO(audio_bytes))

    # Convert to tensor
    wav_tensor = torch.tensor(wav).float()

    # Run VAD
    speech_timestamps = get_speech_timestamps(wav_tensor, model, sampling_rate=sr)

    # Prepare response
    return {
        "speech_segments": speech_timestamps,
        "total_speech_duration_sec": sum([seg["end"] - seg["start"] for seg in speech_timestamps]) / sr
    }
