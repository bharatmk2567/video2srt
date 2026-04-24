# video2srt

A command-line tool for transcribing video/audio files to SRT subtitles using OpenAI Whisper locally on your device. No API calls, no internet required after model download.

## Prerequisites

- Node.js 18+
- ffmpeg (for video files)

Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (with chocolatey)
choco install ffmpeg
```

## Installation

```bash
cd cli
npm install
```

## Usage

```bash
node video2srt.mjs /path/to/video.mp4
```

## Options

```bash
node video2srt.mjs video.mp4 --output subtitles.srt --model base --language en --max-chars 80
```

| Option | Description | Default |
|--------|-------------|---------|
| `--output`, `-o` | Output SRT file path | Same as input with `.srt` extension |
| `--model`, `-m` | Model: tiny, base, small, medium, large, large-v3 | `base` |
| `--language`, `-l` | Language code (e.g., en, es, fr) | Auto-detect |
| `--max-chars` | Maximum characters per subtitle cue | `80` |
| `--pause-threshold` | Min pause in seconds to split cues | `0.2` |
| `--quiet`, `-q` | Suppress progress output | Verbose |

## Examples

```bash
# Basic usage
node video2srt.mjs interview.mp4

# High accuracy with large model
node video2srt.mjs podcast.mp3 --model large --output podcast.srt

# Spanish video with shorter cues
node video2srt.mjs video.mp4 --language es --max-chars 60

# Quiet mode
node video2srt.mjs video.mp4 --quiet
```

## How It Works

1. **Extract Audio** from video files using ffmpeg (if needed)
2. **Decode Audio** to Float32Array using wavefile
3. **Transcribe** using ONNX Whisper models via @huggingface/transformers
4. **Group Words** into subtitle cues using intelligent algorithms:
   - **Pause Detection**: Split at natural pauses (> 0.2s by default)
   - **Character Limit**: Ensure no cue exceeds the max character count
   - **Proportional Splitting**: When a group is too long, split it proportionally
5. **Generate SRT** file with standard formatting

## Model Sizes

| Model | Size | Speed | Notes |
|-------|------|-------|-------|
| tiny | 39M | Fastest | Quick results, lower accuracy |
| base | 74M | Fast | Good balance (default) |
| small | 244M | Medium | Better accuracy |
| medium | 769M | Slow | High accuracy |
| large | 1550M | Slowest | Best accuracy |

**Recommendation**: Start with `base`. Use `small` or `medium` for better accuracy. Use `large` for best quality.

## License

BSD-3-Clause
