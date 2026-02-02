# Qwen3-TTS Streaming

Real-time streaming audio generation for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

## Features

From [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming):
- `stream_generate_voice_clone()` - streaming with voice cloning
- `stream_generate_pcm()` - real-time PCM audio streaming
- `torch.compile` + CUDA graphs optimization
- Crossfade overlap for seamless chunk transitions

Added in this fork:
- **Two-phase streaming** - faster first-chunk latency

## Two-Phase Streaming

Standard streaming with Qwen's TTS library waits for `emit_every_frames` (e.g., 12) before emitting the first audio. Two-phase uses aggressive settings for the first chunk to improve latency, then switches to stable settings.

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 (First N frames)      │  PHASE 2 (Rest of audio)      │
│  - emit_every = 5 (fast)       │  - emit_every = 12 (stable)   │
│  - decode_window = 48          │  - decode_window = 80         │
│  - optimized = OFF             │  - optimized = ON             │
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

## Usage

```python
for chunk, sr in model.stream_generate_voice_clone(
    text="Hello!",
    language="en",
    voice_clone_prompt=prompt,
    # Phase 2 settings
    emit_every_frames=12,
    decode_window_frames=80,
    # Phase 1 settings (two-phase)
    first_chunk_emit_every=5,
    first_chunk_decode_window=48,
    first_chunk_frames=48,
):
    play_audio(chunk, sr)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emit_every_frames` | 8 | Emit audio every N frames |
| `decode_window_frames` | 80 | Decoder context window |
| `overlap_samples` | 512 | Crossfade overlap between chunks |
| `first_chunk_emit_every` | 0 | Phase 1 emit interval (0 = disabled) |
| `first_chunk_decode_window` | 48 | Phase 1 decode window |
| `first_chunk_frames` | 48 | Switch to phase 2 after N frames |

## Installation

```bash
sudo apt install sox
pip install torch torchaudio flash-attn
pip install -e .
```

---

Based on:
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming)
