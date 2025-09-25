from fastapi import FastAPI, UploadFile, HTTPException
import torch
import soundfile as sf
import torchaudio
import io
import numpy as np

TARGET_SAMPLE_RATE = 16000

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
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        audio_buffer = io.BytesIO(data)

        wav = None
        sr = None

        try:
            wav, sr = sf.read(audio_buffer, dtype='float32')
        except RuntimeError:
            # Fallback to torchaudio for formats like MP3 that SoundFile cannot decode
            audio_buffer.seek(0)
            try:
                wav_tensor, sr = torchaudio.load(audio_buffer)
            except Exception as inner_exc:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported or corrupted audio file. Provide WAV/MP3/FLAC."
                ) from inner_exc

            # Convert torchaudio tensor (channels, frames) to mono numpy array
            if wav_tensor.size(0) > 1:
                wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
            wav = wav_tensor.squeeze(0).numpy()
        else:
            # Reset buffer for consistency
            audio_buffer.seek(0)
            if wav.ndim > 1:
                wav = np.mean(wav, axis=1)

        if sr is None:
            raise HTTPException(status_code=400, detail="Could not determine sample rate.")

        wav_tensor = torch.from_numpy(wav).float()

        if wav_tensor.ndim != 1:
            wav_tensor = wav_tensor.flatten()

        if sr != TARGET_SAMPLE_RATE:
            wav_tensor = torchaudio.functional.resample(
                wav_tensor.unsqueeze(0), sr, TARGET_SAMPLE_RATE
            ).squeeze(0)
            sr = TARGET_SAMPLE_RATE

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

    except HTTPException:
        raise
    except Exception as e:
        # Return helpful error for debugging
        raise HTTPException(status_code=500, detail=str(e))
