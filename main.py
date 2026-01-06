from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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

# Global model variables
device = None
models = {}

# Language configuration
LANGUAGE_CONFIG = {
    "hindi": {
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["hindi_male", "hindi_female"],
        "romanization": lambda text: transliterate.process('Devanagari', 'ISO', text),
        "default_speaker": "hindi_male"
    },
    "malayalam": {
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["malayalam_male", "malayalam_female"],
        "romanization": lambda text: transliterate.process('Malayalam', 'ISO', text),
        "default_speaker": "malayalam_male"
    },
    "manipuri": {
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["manipuri_female"],
        "romanization": lambda text: transliterate.process('Bengali', 'ISO', text),
        "default_speaker": "manipuri_female"
    },
    "bengali": {
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["bengali_male", "bengali_female"],
        "romanization": lambda text: transliterate.process('Bengali', 'ISO', text),
        "default_speaker": "bengali_male"
    },
    "rajasthani": {
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["rajasthani_female"],
        "romanization": lambda text: transliterate.process('Devanagari', 'ISO', text),
        "default_speaker": "rajasthani_female"
    },
    "tamil": {
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["tamil_male", "tamil_female"],
        "romanization": lambda text: transliterate.process('Tamil', 'ISO', text, pre_options=['TamilTranscribe']),
        "default_speaker": "tamil_male"
    },
    "telugu": {
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["telugu_male", "telugu_female"],
        "romanization": lambda text: transliterate.process('Telugu', 'ISO', text),
        "default_speaker": "telugu_male"
    },
    "gujarati": {
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["gujarati_male", "gujarati_female"],
        "romanization": lambda text: transliterate.process('Gujarati', 'ISO', text),
        "default_speaker": "gujarati_male"
    },
    "kannada": {
        "model_lang": "indic",
        "model_speaker": "v4_indic",
        "speakers": ["kannada_male", "kannada_female"],
        "romanization": lambda text: transliterate.process('Kannada', 'ISO', text),
        "default_speaker": "kannada_male"
    },
    "english": {
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
        "romanization": None,  # No romanization needed for English
        "default_speaker": "en_0"
    }
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global device, models
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on device: {device}")
    
    # Load unique models (indic and english)
    unique_models = {}
    for lang, config in LANGUAGE_CONFIG.items():
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
    
    # Map languages to their models
    for lang, config in LANGUAGE_CONFIG.items():
        model_key = f"{config['model_lang']}_{config['model_speaker']}"
        models[lang] = unique_models[model_key]
    
    print(f"All models loaded. Supported languages: {list(models.keys())}")
    
    yield
    
    # Shutdown
    print("Shutting down...")
    for model in unique_models.values():
        del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

app = FastAPI(
    title="Multi-Language Silero TTS API",
    description="Text-to-Speech API supporting Indian languages and English",
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

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "device": str(device),
        "supported_languages": list(LANGUAGE_CONFIG.keys()),
        "models_loaded": len(models)
    }

@app.get("/languages")
async def get_languages():
    """Get list of supported languages and their speakers"""
    language_info = {}
    for lang, config in LANGUAGE_CONFIG.items():
        language_info[lang] = {
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
            detail=f"Language '{language}' not supported. Available: {list(LANGUAGE_CONFIG.keys())}"
        )
    
    config = LANGUAGE_CONFIG[language]
    return {
        "language": language,
        "speakers": config["speakers"],
        "default_speaker": config["default_speaker"]
    }

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech in multiple languages
    
    Args:
        text: Text in the specified language (required)
        language: Language code (required) - hindi, tamil, english, etc.
        speaker: Speaker voice (optional, uses default if not specified)
        sample_rate: Audio sample rate (optional, default: 48000)
    
    Returns:
        Audio file in WAV format
    """
    # Validate language
    if request.language not in LANGUAGE_CONFIG:
        raise HTTPException(
            status_code=400, 
            detail=f"Language '{request.language}' not supported. Available: {list(LANGUAGE_CONFIG.keys())}"
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
            detail=f"Speaker '{speaker}' not available for {request.language}. Available: {config['speakers']}"
        )
    
    try:
        # Romanize text if needed (not for English)
        if config["romanization"]:
            processed_text = config["romanization"](request.text)
            print(f"Original: {request.text}")
            print(f"Romanized: {processed_text}")
        else:
            processed_text = request.text
        
        # Generate audio
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
        "language": "hindi",  // required - any supported language
        "speaker": "hindi_male",  // optional, uses default if not provided
        "sample_rate": 48000  // optional, default is 48000
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
                
                if not text.strip():
                    await websocket.send_json({"error": "Text cannot be empty"})
                    continue
                
                if not language:
                    await websocket.send_json({
                        "error": "Language is required",
                        "available_languages": list(LANGUAGE_CONFIG.keys())
                    })
                    continue
                
                if language not in LANGUAGE_CONFIG:
                    await websocket.send_json({
                        "error": f"Language '{language}' not supported",
                        "available_languages": list(LANGUAGE_CONFIG.keys())
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
                        "error": f"Speaker '{speaker}' not available for {language}",
                        "available_speakers": config["speakers"]
                    })
                    continue
                
                print(f"Processing {language}: {text}")
                
                # Romanize text if needed
                if config["romanization"]:
                    processed_text = config["romanization"](text)
                    print(f"Romanized: {processed_text}")
                else:
                    processed_text = text
                
                # Generate audio
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
                print(f"Sent {len(audio_bytes)} bytes of audio for {language}")
                
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9010, reload=True)