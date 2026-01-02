from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
from silero import silero_tts
from aksharamukha import transliterate
import soundfile as sf
import io
import numpy as np

# Global model variables
device = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup: Load the model
    global device, model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device: {device}")
    
    # Load Indic model
    model, _ = silero_tts(
        language="indic",
        speaker="v4_indic"
    )
    model.to(device)
    print("Model loaded successfully!")
    
    yield  # Application runs here
    
    # Shutdown: Clean up resources (optional)
    print("Shutting down...")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

app = FastAPI(
    title="Silero Hindi TTS API",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class TTSRequest(BaseModel):
    text: str
    speaker: str = "hindi_male"  # Default speaker
    sample_rate: int = 48000      # Default sample rate

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "device": str(device),
        "available_speakers": ["hindi_male", "hindi_female"]
    }

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert Hindi text to speech
    
    Args:
        text: Hindi text in Devanagari script
        speaker: Speaker voice (default: hindi_male)
        sample_rate: Audio sample rate (default: 48000)
    
    Returns:
        Audio file in WAV format
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Romanize Hindi text to ISO
        roman_text = transliterate.process(
            "Devanagari",
            "ISO",
            request.text
        )
        print(f"Romanized text: {roman_text}")
        
        # Generate audio
        audio = model.apply_tts(
            text=roman_text,
            speaker=request.speaker,
            sample_rate=request.sample_rate
        )
        
        # Convert to numpy array if it's a tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Create in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio, request.sample_rate, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=output.wav"
            }
        )
        
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

@app.get("/speakers")
async def get_speakers():
    """Get list of available speakers"""
    return {
        "speakers": ["hindi_male", "hindi_female"],
        "default": "hindi_male"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)