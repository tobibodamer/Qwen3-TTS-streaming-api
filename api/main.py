import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

# Set precision before any CUDA operations
torch.set_float32_matmul_precision('high')

from qwen_tts import Qwen3TTSModel
from fastapi.responses import StreamingResponse
import io
import soundfile as sf
from typing import Optional, Dict
import os
import traceback
import numpy as np
import threading
import queue

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

logger = logging.getLogger("uvicorn.error")
logger.propagate = False

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str
    response_format: Optional[str] = "wav"
    language: Optional[str] = None

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = os.getenv("MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
VOICES_DIR = os.getenv("VOICES_DIR", "voices")

tts_model = Qwen3TTSModel.from_pretrained(
    model_path,
    device_map=device,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

tts_model.enable_streaming_optimizations(
    decode_window_frames=80,
    use_compile=True,
    compile_mode="reduce-overhead",
)

# In-memory cache for voice prompts
voice_cache: Dict[str, dict] = {}
model_ready = False
    
# Warmup: run dummy generation to initialize torch.compile and CUDA graphs
def warmup_model(prompt):
    logger.info("Warming up TTS model...")
    for _ in tts_model.stream_generate_voice_clone(
        # Reference text must be longer to properly initialise model
        text="A rainbow is a meteorological phenomenon that is caused by reflection, refraction and dispersion of light in water droplets.",
        language="english",
        voice_clone_prompt=prompt,
        overlap_samples=512,
        emit_every_frames=12,
        decode_window_frames=80,
        first_chunk_emit_every=5,
        first_chunk_decode_window=48,
        first_chunk_frames=48,
    ):
        pass  # Discard output
    logger.info("TTS model ready")

def get_voice_prompt(voice_name: str):
    if not os.path.exists(VOICES_DIR):
        os.makedirs(VOICES_DIR, exist_ok=True)
    
    if voice_name in voice_cache:
        return voice_cache[voice_name]
    
    wav_path = os.path.join(VOICES_DIR, f"{voice_name}.wav")
    txt_path = os.path.join(VOICES_DIR, f"{voice_name}.txt")
    
    if not os.path.exists(wav_path):
        # Fallback to project root if voices dir doesn't have it (e.g. for the default kuklina-1)
        if os.path.exists(f"{voice_name}.wav"):
            wav_path = f"{voice_name}.wav"
            txt_path = f"{voice_name}.txt"
        else:
            raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found.")
    
    ref_text = None
    x_vector_only = True
    
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
            x_vector_only = False
    
    prompt = tts_model.create_voice_clone_prompt(
        ref_audio=wav_path,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only
    )
    
    voice_cache[voice_name] = prompt
    return prompt

@asynccontextmanager
async def lifespan(app: FastAPI):
    voice = next(iter(_list_voices()), None)
    if voice:
        warmup_model(get_voice_prompt(voice))

    global model_ready
    # The model is loaded at module level currently, 
    # so if we reach here, it's mostly ready.
    model_ready = True
    yield
    
app = FastAPI(lifespan=lifespan)

def audio_stream_generator(text, language, voice_clone_prompt, response_format):
    print(f"Starting audio generation for text: {text[:50]}...")
    
    audio_queue = queue.Queue()
    sample_rate_holder = [None]  # To capture sample rate from producer
    
    def producer():
        """Generate audio chunks into queue."""
        try:
            for audio_chunk, sample_rate in tts_model.stream_generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                overlap_samples=512,
                # Phase 2 settings (stable)
                emit_every_frames=12,
                decode_window_frames=80,
                # Phase 1 settings (fast first chunk)
                first_chunk_emit_every=5,
                first_chunk_decode_window=48,
                first_chunk_frames=48,
            ):
                if sample_rate_holder[0] is None:
                    sample_rate_holder[0] = sample_rate
                audio_queue.put(audio_chunk)
        except Exception as e:
            logger.exception(f"TTS generation error for text '{text[:50]}': {type(e).__name__}: {e}")
        finally:
            audio_queue.put(None)  # Sentinel to signal completion

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    fmt = response_format.lower()
    header_sent = False
    
    while True:
        try:
            item = audio_queue.get(timeout=30)  # Wait up to 30s for a chunk
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            
            chunk: np.ndarray = item
            sr = sample_rate_holder[0]
            buffer = io.BytesIO()
            
            logger.debug(f"Generated chunk of length {chunk.size}")
            
            if fmt == "wav":
                if not header_sent:
                    sf.write(buffer, chunk, sr, format="WAV", subtype="PCM_16")
                    header_sent = True
                    yield buffer.getvalue()
                else:
                    pcm_data = (chunk * 32767).astype(np.int16)
                    yield pcm_data.tobytes()
            elif fmt in ["pcm", "raw"]:
                pcm_data = (chunk * 32767).astype(np.int16)
                yield pcm_data.tobytes()
            else:
                sf.write(buffer, chunk, sr, format=fmt.upper())
                yield buffer.getvalue()
                
        except queue.Empty:
            logger.warning("Timed out waiting for audio chunk")
            break
        except Exception as e:
            logger.error(f"ERROR in consumer: {str(e)}")
            break

    logger.info("Generation complete.")
    
def _list_voices():
    voices: list[str] = []
    if os.path.exists(VOICES_DIR):
        for f in os.listdir(VOICES_DIR):
            if f.endswith(".wav"):
                voices.append(f[:-4])
    return voices

@app.get("/health")
def health_check():
    if model_ready:
        return {"status": "ready"}
    else:
        return {"status": "loading"}, 503

@app.get("/v1/voices")
def list_voices():
    voices = _list_voices()
    return {"voices": voices}

@app.post("/v1/audio/speech")
def create_speech(request: SpeechRequest):
    logger.debug(f"Received speech request: {request}")
    try:
        voice_prompt = get_voice_prompt(request.voice)
    except HTTPException as e:
        logger.error(f"HTTP Error: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        audio_stream_generator(request.input, request.language, voice_prompt, request.response_format),
        media_type=f"audio/{request.response_format}"
    )
