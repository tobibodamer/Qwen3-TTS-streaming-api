import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
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
import numpy as np
import threading
import queue

logger = logging.getLogger("uvicorn.error")

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str
    response_format: Optional[str] = "pcm" # Default to raw PCM for streaming
    language: Optional[str] = None

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = os.getenv("MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
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

sem = asyncio.Semaphore(1)
    
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

async def audio_stream_generator(text, language, voice_clone_prompt, response_format, request: Request):
    async with sem:
        print(f"Starting audio generation for text: {text[:50]}...")
    
        audio_queue = queue.Queue()
        sample_rate_holder = [None]
        stop_event = threading.Event()  # Signal to stop generation
        
        def producer():
            """Generate audio chunks into queue (runs in background thread)."""
            try:
                for audio_chunk, sample_rate in tts_model.stream_generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt,
                    overlap_samples=512,
                    emit_every_frames=12,
                    decode_window_frames=80,
                    first_chunk_emit_every=5,
                    first_chunk_decode_window=48,
                    first_chunk_frames=48,
                ):
                    # Check if we should stop
                    if stop_event.is_set():
                        logger.info("Stop signal received, halting TTS generation.")
                        break
                    
                    if sample_rate_holder[0] is None:
                        sample_rate_holder[0] = sample_rate
                    
                    audio_queue.put((audio_chunk, sample_rate))
                    
            except Exception as e:
                logger.exception(f"TTS generation error: {e}")
                audio_queue.put(e)
            finally:
                audio_queue.put(None)  # Sentinel
        
        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        
        fmt = response_format.lower()
        header_sent = False
        
        try:
            while True:
                # Check for disconnect
                if await request.is_disconnected():
                    logger.info("Client disconnected, signaling stop.")
                    stop_event.set()  # Signal the producer to stop
                    break
                
                try:
                    item = audio_queue.get(timeout=0.1)
                    
                    if item is None:  # Generation complete
                        break
                        
                    if isinstance(item, Exception):
                        raise item
                    
                    audio_chunk, sample_rate = item
                    chunk: np.ndarray = audio_chunk
                    sr = sample_rate
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
                    continue
                    
        except Exception as e:
            logger.error(f"ERROR in consumer: {str(e)}")
            stop_event.set()  # Signal stop on any error
            raise
        finally:
            # Ensure we always signal stop and wait for thread cleanup
            stop_event.set()
            thread.join(timeout=2.0)  # Wait up to 2s for clean shutdown
            
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
async def create_speech(request: SpeechRequest, r: Request):
    logger.debug(f"Received speech request: {request}")
    
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is not ready")
    
    if request.response_format not in ["pcm", "raw"]:
        raise HTTPException(status_code=400, detail="Unsupported response format. Use 'pcm' for streaming.")
    
    try:
        voice_prompt = get_voice_prompt(request.voice)
    except HTTPException as e:
        logger.error(f"HTTP Error: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        audio_stream_generator(
            request.input, 
            request.language, 
            voice_prompt, 
            request.response_format, 
            r
        ),
        media_type=f"audio/{request.response_format}"
    )
