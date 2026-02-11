# Qwen3-TTS Streaming

Real-time streaming audio generation for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

## Features

From [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming):
- `stream_generate_voice_clone()` - streaming with voice cloning
- `stream_generate_pcm()` - real-time PCM audio streaming
- `torch.compile` + CUDA graphs optimization

Added in this fork:
- **Two-phase streaming** - faster first-chunk latency
- **Multiple EOS token detection** - broader termination coverage for reliable generation stopping. Fixes sped-up audio and runaway generation in streaming
- **Hann window crossfade** - click-free chunk boundaries with proper fade-in/fade-out
- **Batch generation** - process multiple texts in a single forward pass with `generate_voice_clone(text=List[str])` for ~2.6x speedup over sequential
- **Repetition penalty for streaming** - prevents token loops that cause looping audio and runaway generation. Defaults to 1.0 (disabled) because streaming generates frame-by-frame with CUDA graph constraints where repetition manifests differently than the non-streaming path (which defaults to 1.05)

## Installation

```bash
sudo apt install sox
pip install torch torchaudio flash-attn
pip install -e .
```

## Usage

```python
import torch
import sounddevice as sd
from qwen_tts import Qwen3TTSModel

# Load model
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-Base",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# Enable optimizations (recommended for streaming)
model.enable_streaming_optimizations(
    decode_window_frames=80,
    use_compile=True,
    compile_mode="reduce-overhead",
)

# Create voice clone prompt from reference audio
prompt = model.create_voice_clone_prompt(
    ref_audio="reference.wav",
    ref_text="Transcript of the reference audio.",
)

# Stream audio with two-phase settings
for chunk, sr in model.stream_generate_voice_clone(
    text="Hello, this is a streaming TTS demo!",
    language="en",
    voice_clone_prompt=prompt,
    # Phase 2 settings (stable)
    emit_every_frames=12,
    decode_window_frames=80,
    # Phase 1 settings (fast first chunk)
    first_chunk_emit_every=5,
    first_chunk_decode_window=48,
    first_chunk_frames=48,
):
    sd.play(chunk, sr)
    sd.wait()
```

## Batch Generation

Generate audio for multiple texts in a single forward pass. Pass a list of strings to `generate_voice_clone()` — a single voice prompt broadcasts to all items automatically.

```python
import soundfile as sf

texts = [
    (
        "The development of artificial intelligence has transformed nearly every aspect of "
        "modern life, from the way we communicate with one another to the fundamental nature "
        "of work itself. Machine learning algorithms now power recommendation systems that "
        "curate our news feeds, voice assistants that manage our daily schedules, and "
        "autonomous vehicles that navigate complex urban environments. Yet despite these "
        "remarkable advances, the field continues to grapple with profound questions about "
        "bias, transparency, and the ethical implications of delegating critical decisions "
        "to automated systems that operate in ways their creators do not fully understand."
    ),

    (
        "Throughout history, technological revolutions have always been accompanied by "
        "periods of significant social disruption and adaptation. The industrial revolution "
        "displaced millions of agricultural workers, but eventually created entirely new "
        "categories of employment that no one could have predicted. Similarly, the digital "
        "revolution of the late twentieth century eliminated countless clerical and "
        "manufacturing positions while simultaneously giving rise to the software industry, "
        "e-commerce, and the gig economy. The current wave of artificial intelligence "
        "promises to follow a comparable pattern, though the speed and scale of disruption "
        "may exceed anything we have experienced before."
    ),

    (
        "Looking ahead, the most transformative applications of artificial intelligence "
        "are likely to emerge not from any single breakthrough, but from the convergence of "
        "multiple technologies working in concert. Large language models combined with "
        "robotics could revolutionize healthcare delivery in underserved communities. "
        "Computer vision paired with satellite imagery and climate models could provide "
        "early warning systems for natural disasters with unprecedented accuracy. And "
        "generative AI tools are already enabling artists, musicians, and writers to explore "
        "creative possibilities that were previously unimaginable, raising fascinating "
        "questions about the nature of authorship and artistic expression in an age of "
        "human-machine collaboration."
    ),
]

# Single call returns a list of (wav_array, sample_rate) tuples
results = model.generate_voice_clone(
    text=texts,
    language="en",
    voice_clone_prompt=prompt,       # broadcast to all items
)

for i, (wav, sr) in enumerate(results):
    sf.write(f"output_{i}.wav", wav, sr)
```

**Benchmarks** (Qwen3-TTS-12Hz-1.7B-Base, RTX 5060 Ti 16GB, 3 paragraphs, emit every 12 frames, decode window 80 frames):

| Metric | Sequential | Batch | Speedup |
|--------|-----------|-------|---------|
| Total time | 54.06s | 20.44s | **2.64x** |
| Audio generated | 104.80s | 106.24s | — |
| Throughput | 1.94x | 5.20x | — |
| RTF | 0.516 | 0.192 | — |

Batch streaming (`batch_stream_generate_voice_clone()`) is also available for incremental chunk delivery, but has higher TTFB (~8.6s) due to lockstep prefill — prefer non-streaming batch for offline/buffered workloads.

## Streaming Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emit_every_frames` | 8 | Emit audio every N frames |
| `decode_window_frames` | 80 | Decoder context window |
| `overlap_samples` | 512 | Crossfade overlap between chunks (0 to disable) |
| `max_frames` | 10000 | Maximum codec frames to generate |
| `first_chunk_emit_every` | 0 | Phase 1 emit interval (0 = disabled) |
| `first_chunk_decode_window` | 48 | Phase 1 decode window |
| `first_chunk_frames` | 48 | Switch to phase 2 after N frames |
| `repetition_penalty` | 1.0 | Penalizes repeated tokens (1.0 = disabled) |
| `repetition_penalty_window` | 100 | Only penalize tokens from the last N steps (0 = unlimited) |

## Two-Phase Streaming

Standard streaming with Qwen's TTS library waits for `emit_every_frames` (e.g., 12) before emitting the first audio. Two-phase uses aggressive settings for the first chunk to improve latency, then switches to stable settings.

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 (First N frames)      │  PHASE 2 (Rest of audio)      │
│  - emit_every = 5 (fast)       │  - emit_every = 12 (stable)   │
│  - decode_window = 48          │  - decode_window = 80         │
│  → FAST first chunk            │  → QUALITY for rest           │
└─────────────────────────────────────────────────────────────────┘
```

### Benchmarks

| Test | Method | emit | 1st Chunk | 1st Spdup | Total | Tot Spdup | RTF |
|------|--------|------|-----------|-----------|-------|-----------|-----|
| 2 | Baseline (no opt) | 12 | 570ms | 1.00x | 3.16s | 1.00x | 0.56 |
| 3 | Optimized | 12 | 389ms | 1.47x | 2.37s | 1.34x | 0.37 |
| 4 | Optimized_2 (stable) | 12 | 382ms | 1.49x | 2.27s | 1.39x | 0.36 |
| 5 | **Two-phase (5→12)** | 5→12 | **208ms** | **2.75x** | 2.58s | 1.23x | 0.39 |

User hears audio **362ms earlier** vs baseline, **174ms earlier** vs only optimized.

**First-chunk latency improvement:**
- vs Baseline: **2.75x faster** (570ms → 208ms, saves 362ms)
- vs Optimized: **1.87x faster** (389ms → 208ms, saves 181ms)
- vs Optimized_2: **1.84x faster** (382ms → 208ms, saves 174ms)

## Audio Quality Fixes

Streaming TTS can produce clicks, pops, and artifacts at chunk boundaries. This fork implements several fixes:

### Crossfade Blending

Chunks are blended using a Hann window crossfade to eliminate boundary discontinuities:

```python
# ~21ms at 24kHz, matches RMS check window
# Lower values may cause clicks, set to 0 to disable
DEFAULT_BLEND_SAMPLES = 512

# Hann crossfade
fade_out = 0.5 * (1 + np.cos(np.pi * t))
fade_in = 0.5 * (1 - np.cos(np.pi * t))
blended = prev_tail * fade_out + curr_head * fade_in
```

### Overlap Trimming

Each chunk is processed in this order to prevent audio duplication (echo artifacts):

1. Crossfade current chunk's HEAD with previous chunk's saved TAIL
2. Apply fade-in (first chunk only)
3. Save FULL processed chunk for next iteration's crossfade
4. Trim END of chunk before emission (this region will be replaced by next chunk's crossfade)
5. Yield trimmed chunk

### First/Last Chunk Fades

- **First chunk**: Hann fade-in prevents pop at audio start
- **Final chunk**: Hann fade-out prevents pop at audio end

## Optimization API

### enable_streaming_optimizations()

Call after loading the model to enable torch.compile and CUDA graphs:

```python
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# Enable optimizations (recommended)
model.enable_streaming_optimizations(
    decode_window_frames=80,         # Must match streaming parameter
    use_compile=True,                # torch.compile the decoder
    compile_mode="reduce-overhead",  # Includes CUDA graphs automatically
)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `decode_window_frames` | 80 | Window size (must match streaming call) |
| `use_compile` | True | Apply torch.compile to decoder |
| `use_cuda_graphs` | True | Capture CUDA graphs for fixed window |
| `compile_mode` | "reduce-overhead" | torch.compile mode |
| `use_fast_codebook` | False | Use fast codebook generation (experimental) |
| `compile_codebook_predictor` | True | Apply torch.compile to codebook predictor |

---

Based on:
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming)
