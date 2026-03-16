from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional
import torch
from silero import silero_tts
from aksharamukha import transliterate
import soundfile as sf
import io
import json
import base64
import numpy as np
from scipy.signal import resample_poly
from text_normalizer import TTSTextNormalizer  # Import the normalizer

# Global model variables
device = None
models = {}
normalizer = None  # Add global normalizer

# STT: language code -> (model, decoder, utils) from PyTorch Hub
stt_models = {}

# Language configuration with ISO 639-1 codes
LANGUAGE_CONFIG = {
    "hi": {  # Hindi
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["hindi_male", "hindi_female"],
        "romanization": lambda text: transliterate.process('Devanagari', 'ISO', text),
        "default_speaker": "hindi_male",
        "name": "Hindi"
    },
    "ml": {  # Malayalam
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["malayalam_male", "malayalam_female"],
        "romanization": lambda text: transliterate.process('Malayalam', 'ISO', text),
        "default_speaker": "malayalam_male",
        "name": "Malayalam"
    },
    "mni": {  # Manipuri
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["manipuri_female"],
        "romanization": lambda text: transliterate.process('Bengali', 'ISO', text),
        "default_speaker": "manipuri_female",
        "name": "Manipuri"
    },
    "bn": {  # Bengali
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["bengali_male", "bengali_female"],
        "romanization": lambda text: transliterate.process('Bengali', 'ISO', text),
        "default_speaker": "bengali_male",
        "name": "Bengali"
    },
    "raj": {  # Rajasthani
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["rajasthani_female"],
        "romanization": lambda text: transliterate.process('Devanagari', 'ISO', text),
        "default_speaker": "rajasthani_female",
        "name": "Rajasthani"
    },
    "ta": {  # Tamil
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["tamil_male", "tamil_female"],
        "romanization": lambda text: transliterate.process('Tamil', 'ISO', text, pre_options=['TamilTranscribe']),
        "default_speaker": "tamil_male",
        "name": "Tamil"
    },
    "te": {  # Telugu
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["telugu_male", "telugu_female"],
        "romanization": lambda text: transliterate.process('Telugu', 'ISO', text),
        "default_speaker": "telugu_male",
        "name": "Telugu"
    },
    "gu": {  # Gujarati
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["gujarati_male", "gujarati_female"],
        "romanization": lambda text: transliterate.process('Gujarati', 'ISO', text),
        "default_speaker": "gujarati_male",
        "name": "Gujarati"
    },
    "kn": {  # Kannada
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["kannada_male", "kannada_female"],
        "romanization": lambda text: transliterate.process('Kannada', 'ISO', text),
        "default_speaker": "kannada_male",
        "name": "Kannada"
    },
    "en": {  # English
        "model_lang": "en",
        "model_speaker": "v3_en",
        "speakers": ["en_0", "en_1", "en_2", "en_3", "en_4", "en_5", "en_6", "en_7", "en_8", "en_9", 
                     "en_10", "en_11", "en_12", "en_13", "en_14", "en_15", "en_16", "en_17", "en_18", 
                     "en_19", "en_20", "en_21", "en_22", "en_23", "en_24", "en_25", "en_26", "en_27", 
                     "en_28", "en_29", "en_30", "en_31", "en_32", "en_33", "en_34", "en_35", "en_36", 
                     "en_37", "en_38", "en_39", "en_40", "en_41", "en_42", "en_43", "en_44", "en_45", 
                     "en_46", "en_47", "en_48", "en_49", "en_50", "en_51", "en_52", "en_53", "en_54", 
                     "en_55", "en_56", "en_57", "en_58", "en_59", "en_60", "en_61", "en_62", "en_63", 
                     "en_64", "en_65", "en_66", "en_67", "en_68", "en_69", "en_70", "en_71", "en_72", 
                     "en_73", "en_74", "en_75", "en_76", "en_77", "en_78", "en_79", "en_80", "en_81", 
                     "en_82", "en_83", "en_84", "en_85", "en_86", "en_87", "en_88", "en_89", "en_90", 
                     "en_91", "en_92", "en_93", "en_94", "en_95", "en_96", "en_97", "en_98", "en_99", 
                     "en_100", "en_101", "en_102", "en_103", "en_104", "en_105", "en_106", "en_107", 
                     "en_108", "en_109", "en_110", "en_111", "en_112", "en_113", "en_114", "en_115", 
                     "en_116", "en_117"],
        "romanization": None,
        "default_speaker": "en_0",
        "name": "English"
    }
}

# STT language config: ISO-style code -> Silero STT language name (from latest_silero_models.yml)
STT_LANGUAGE_CONFIG = {
    "en": {"name": "English"},
    "de": {"name": "German"},
    "es": {"name": "Spanish"},
    "ua": {"name": "Ukrainian"},
}

# Silero STT expects 16 kHz
STT_SAMPLE_RATE = 16000

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global device, models, normalizer, stt_models
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on device: {device}")
    
    # Initialize text normalizer
    normalizer = TTSTextNormalizer()
    print("Text normalizer initialized")
    
    # Load unique models (indic and english)
    unique_models = {}
    for lang_code, config in LANGUAGE_CONFIG.items():
        model_key = f"{config['model_lang']}_{config['model_speaker']}"
        if model_key not in unique_models:
            print(f"Loading model: {model_key}")
            model, _ = silero_tts(
                language=config['model_lang'],
                speaker=config['model_speaker']
            )
            model.to(device)
            unique_models[model_key] = model
            print(f"Model {model_key} loaded successfully!")
    
    # Map language codes to their models
    for lang_code, config in LANGUAGE_CONFIG.items():
        model_key = f"{config['model_lang']}_{config['model_speaker']}"
        models[lang_code] = unique_models[model_key]
    
    print(f"All models loaded. Supported language codes: {list(models.keys())}")
    
    # Load STT models (PyTorch Hub)
    for lang_code in STT_LANGUAGE_CONFIG:
        try:
            print(f"Loading STT model: {lang_code}")
            model, decoder, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_stt",
                language=lang_code,
                device=device,
            )
            model.eval()
            stt_models[lang_code] = (model, decoder, utils)
            print(f"STT model {lang_code} loaded successfully!")
        except Exception as e:
            print(f"Failed to load STT model {lang_code}: {e}")
    print(f"STT models loaded: {list(stt_models.keys())}")
    
    yield
    
    # Shutdown
    print("Shutting down...")
    for model in unique_models.values():
        del model
    stt_models.clear()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

app = FastAPI(
    title="Multi-Language Silero TTS & STT API",
    description="Text-to-Speech and Speech-to-Text API supporting Indian languages, English, and other languages with ISO 639-1 language codes",
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
    language: str
    speaker: Optional[str] = None
    sample_rate: Optional[int] = None
    normalize: Optional[bool] = True  # Enable normalization by default


def _run_stt(audio_bytes: bytes, language: str) -> str:
    """Run Silero STT on audio bytes. Loads with soundfile (no TorchCodec/FFmpeg). Returns decoded text."""
    if language not in stt_models:
        raise ValueError(f"STT language '{language}' not loaded")
    model, decoder, utils = stt_models[language]
    _read_batch, _split_into_batches, _read_audio, prepare_model_input = utils
    # Load with soundfile to avoid torchaudio.load (and TorchCodec/FFmpeg)
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != STT_SAMPLE_RATE:
        # Resample to 16 kHz using scipy (no torchaudio dependency for load path)
        data = resample_poly(data.astype(np.float64), STT_SAMPLE_RATE, sr).astype(np.float32)
        sr = STT_SAMPLE_RATE
    wav = torch.from_numpy(data).float().squeeze()
    batch = [wav]
    model_input = prepare_model_input(batch, device=device)
    with torch.no_grad():
        output = model(model_input)
    return decoder(output[0].cpu())


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "device": str(device),
        "supported_languages": {code: config["name"] for code, config in LANGUAGE_CONFIG.items()},
        "models_loaded": len(models),
        "text_normalization": "enabled",
        "stt_available": len(stt_models) > 0,
        "stt_supported_languages": list(stt_models.keys()),
    }

@app.get("/languages")
async def get_languages():
    """Get list of supported languages and their speakers"""
    language_info = {}
    for lang_code, config in LANGUAGE_CONFIG.items():
        language_info[lang_code] = {
            "name": config["name"],
            "speakers": config["speakers"],
            "default_speaker": config["default_speaker"]
        }
    return language_info

@app.get("/languages/{language}/speakers")
async def get_language_speakers(language: str):
    """Get speakers for a specific language"""
    if language not in LANGUAGE_CONFIG:
        raise HTTPException(
            status_code=404, 
            detail=f"Language code '{language}' not supported. Available: {list(LANGUAGE_CONFIG.keys())}"
        )
    
    config = LANGUAGE_CONFIG[language]
    return {
        "language_code": language,
        "language_name": config["name"],
        "speakers": config["speakers"],
        "default_speaker": config["default_speaker"]
    }


@app.get("/stt/languages")
async def get_stt_languages():
    """Get list of supported STT languages"""
    return {
        code: {"name": config["name"]}
        for code, config in STT_LANGUAGE_CONFIG.items()
        if code in stt_models
    }


@app.post("/stt")
async def speech_to_text(
    language: str = Form(..., description="ISO 639-1 language code (e.g. en, de, es, ua)"),
    audio: UploadFile = File(..., description="Audio file (WAV preferred, 16 kHz recommended)"),
):
    """
    Convert speech to text (transcription).

    Args:
        language: ISO 639-1 language code (required) - en, de, es, ua.
        audio: Audio file upload (required). WAV or other TorchAudio-compatible format; resampled to 16 kHz internally if needed.

    Returns:
        JSON with transcribed text: {"text": "..."}
    """
    if language not in STT_LANGUAGE_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Language code '{language}' not supported. Available: {list(stt_models.keys())}",
        )
    if language not in stt_models:
        raise HTTPException(status_code=503, detail="STT model not loaded for this language")
    if not audio.filename:
        raise HTTPException(status_code=400, detail="Audio file is required")
    try:
        audio_bytes = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {str(e)}")
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")
    try:
        text = _run_stt(audio_bytes, language)
        return {"text": text}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech in multiple languages
    
    Args:
        text: Text in the specified language (required)
        language: ISO 639-1 language code (required) - en, hi, ta, bn, etc.
        speaker: Speaker voice (optional, uses default if not specified)
        sample_rate: Audio sample rate (optional, default: 48000)
        normalize: Enable text normalization for numbers/units (default: True)
    
    Returns:
        Audio file in WAV format
    """
    # Validate language
    if request.language not in LANGUAGE_CONFIG:
        raise HTTPException(
            status_code=400, 
            detail=f"Language code '{request.language}' not supported. Available: {list(LANGUAGE_CONFIG.keys())}"
        )
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    config = LANGUAGE_CONFIG[request.language]
    model = models.get(request.language)
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Use default speaker if none specified
    speaker = request.speaker if request.speaker else config["default_speaker"]
    
    # Use default sample rate if none specified
    sample_rate = request.sample_rate if request.sample_rate else 48000
    
    # Validate speaker
    if speaker not in config["speakers"]:
        raise HTTPException(
            status_code=400,
            detail=f"Speaker '{speaker}' not available for {config['name']} ({request.language}). Available: {config['speakers']}"
        )
    
    try:
        # STEP 1: Normalize text (numbers, units, etc.)
        if request.normalize:
            normalized_text = normalizer.normalize(request.text, request.language)
            print(f"Original: {request.text}")
            print(f"Normalized: {normalized_text}")
        else:
            normalized_text = request.text
        
        # STEP 2: Romanize text if needed (not for English)
        if config["romanization"]:
            processed_text = config["romanization"](normalized_text)
            print(f"Romanized: {processed_text}")
        else:
            processed_text = normalized_text
        
        # STEP 3: Generate audio
        audio = model.apply_tts(
            text=processed_text,
            speaker=speaker,
            sample_rate=sample_rate
        )
        
        # Convert to numpy array if it's a tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Create in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename={request.language}_output.wav"
            }
        )
        
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """
    WebSocket endpoint for streaming text-to-speech
    
    Expected message format:
    {
        "text": "Text in the specified language",  // required
        "language": "hi",  // required - ISO 639-1 code (en, hi, ta, bn, etc.)
        "speaker": "hindi_male",  // optional, uses default if not provided
        "sample_rate": 48000,  // optional, default is 48000
        "normalize": true  // optional, enable text normalization (default: true)
    }
    
    Response format:
    - Binary audio data (WAV format)
    - Or JSON error message: {"error": "error message"}
    """
    await websocket.accept()
    print("WebSocket connection established")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                text = message.get("text", "")
                language = message.get("language")
                speaker = message.get("speaker")
                sample_rate = message.get("sample_rate", 48000)
                normalize = message.get("normalize", True)
                
                if not text.strip():
                    await websocket.send_json({"error": "Text cannot be empty"})
                    continue
                
                if not language:
                    await websocket.send_json({
                        "error": "Language code is required",
                        "available_languages": {code: config["name"] for code, config in LANGUAGE_CONFIG.items()}
                    })
                    continue
                
                if language not in LANGUAGE_CONFIG:
                    await websocket.send_json({
                        "error": f"Language code '{language}' not supported",
                        "available_languages": {code: config["name"] for code, config in LANGUAGE_CONFIG.items()}
                    })
                    continue
                
                config = LANGUAGE_CONFIG[language]
                model = models.get(language)
                
                if model is None:
                    await websocket.send_json({"error": "Model not loaded"})
                    continue
                
                # Use default speaker if none specified
                if not speaker:
                    speaker = config["default_speaker"]
                
                # Validate speaker
                if speaker not in config["speakers"]:
                    await websocket.send_json({
                        "error": f"Speaker '{speaker}' not available for {config['name']} ({language})",
                        "available_speakers": config["speakers"]
                    })
                    continue
                
                print(f"Processing {config['name']} ({language}): {text}")
                
                # STEP 1: Normalize text
                if normalize:
                    normalized_text = normalizer.normalize(text, language)
                    print(f"Normalized: {normalized_text}")
                else:
                    normalized_text = text
                
                # STEP 2: Romanize text if needed
                if config["romanization"]:
                    processed_text = config["romanization"](normalized_text)
                    print(f"Romanized: {processed_text}")
                else:
                    processed_text = normalized_text
                
                # STEP 3: Generate audio
                audio = model.apply_tts(
                    text=processed_text,
                    speaker=speaker,
                    sample_rate=sample_rate
                )
                
                # Convert to numpy array if it's a tensor
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                
                # Create in-memory buffer
                buffer = io.BytesIO()
                sf.write(buffer, audio, sample_rate, format='WAV')
                buffer.seek(0)
                
                # Send audio data
                audio_bytes = buffer.read()
                await websocket.send_bytes(audio_bytes)
                print(f"Sent {len(audio_bytes)} bytes of audio for {config['name']} ({language})")
                
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})
            except Exception as e:
                error_msg = f"Error generating audio: {str(e)}"
                print(error_msg)
                await websocket.send_json({"error": error_msg})
                
    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")


@app.websocket("/ws/stt")
async def websocket_stt(websocket: WebSocket):
    """
    WebSocket endpoint for speech-to-text (transcription).

    Expected message format:
    {
        "audio": "<base64-encoded audio bytes>",  // required
        "language": "en"  // required - ISO 639-1 code (en, de, es, ua)
    }

    Response format:
    - Success: {"text": "transcribed text"}
    - Error: {"error": "error message"} or {"error": "...", "available_languages": [...]}
    """
    await websocket.accept()
    print("WebSocket STT connection established")

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                audio_b64 = message.get("audio", "")
                language = message.get("language")

                if not audio_b64:
                    await websocket.send_json({"error": "Audio (base64) is required"})
                    continue

                if not language:
                    await websocket.send_json({
                        "error": "Language code is required",
                        "available_languages": list(stt_models.keys()),
                    })
                    continue

                if language not in STT_LANGUAGE_CONFIG:
                    await websocket.send_json({
                        "error": f"Language code '{language}' not supported",
                        "available_languages": list(stt_models.keys()),
                    })
                    continue

                if language not in stt_models:
                    await websocket.send_json({"error": "STT model not loaded for this language"})
                    continue

                try:
                    audio_bytes = base64.b64decode(audio_b64)
                except Exception:
                    await websocket.send_json({"error": "Invalid base64 in audio field"})
                    continue

                if not audio_bytes:
                    await websocket.send_json({"error": "Audio data is empty"})
                    continue

                try:
                    text = _run_stt(audio_bytes, language)
                    await websocket.send_json({"text": text})
                    print(f"STT ({language}): sent text length {len(text)}")
                except ValueError as e:
                    await websocket.send_json({"error": str(e)})
                except Exception as e:
                    error_msg = f"Error transcribing audio: {str(e)}"
                    print(error_msg)
                    await websocket.send_json({"error": error_msg})

            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})

    except WebSocketDisconnect:
        print("WebSocket STT connection closed")
    except Exception as e:
        print(f"WebSocket STT error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9010, reload=False)