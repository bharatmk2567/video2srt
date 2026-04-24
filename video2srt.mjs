#!/usr/bin/env node
/**
 * video2srt - CLI tool for transcribing video/audio to SRT subtitles
 * 
 * Uses @huggingface/transformers with ONNX Whisper models
 * Usage: node video2srt.mjs video.mp4
 */

import { pipeline } from "@huggingface/transformers";
import fs from "fs";
import path from "path";
import { createRequire } from "module";
import { execSync } from "child_process";
import wavefilePkg from "wavefile";
const { WaveFile } = wavefilePkg;
import os from "os";

const require = createRequire(import.meta.url);

// Model mapping from simple names to ONNX community models
const MODEL_MAP = {
  "tiny": "onnx-community/whisper-tiny",
  "base": "onnx-community/whisper-base",
  "small": "onnx-community/whisper-small",
  "medium": "onnx-community/whisper-medium",
  "large": "onnx-community/whisper-large-v3",
  "large-v2": "onnx-community/whisper-large-v2",
  "large-v3": "onnx-community/whisper-large-v3",
};

// Parse arguments
const args = process.argv.slice(2);
if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
  console.log(`
Subtitler CLI (Node.js) - Local AI video transcription to SRT

Usage:
  node subtitler-cli-node.mjs <input> [options]

Options:
  --output, -o          Output SRT file path
  --model, -m           Model: tiny, base, small, medium, large, large-v3 (default: base)
  --language, -l        Language code (auto-detect if not specified)
  --max-chars           Maximum characters per cue (default: 80)
  --pause-threshold     Min pause in seconds to split (default: 0.2)
  --quiet, -q           Suppress progress output
  --help, -h            Show this help

Examples:
  node subtitler-cli-node.mjs video.mp4
  node subtitler-cli-node.mjs audio.mp3 --model large --output subtitles.srt
`);
  process.exit(args.includes("--help") || args.includes("-h") ? 0 : 1);
}

function parseArgs() {
  const options = {
    input: args[0],
    output: null,
    model: "base",
    language: null,
    maxChars: 80,
    pauseThreshold: 0.2,
    quiet: false,
  };

  for (let i = 1; i < args.length; i++) {
    switch (args[i]) {
      case "--output":
      case "-o":
        options.output = args[++i];
        break;
      case "--model":
      case "-m":
        options.model = args[++i];
        break;
      case "--language":
      case "-l":
        options.language = args[++i];
        break;
      case "--max-chars":
        options.maxChars = parseInt(args[++i]);
        break;
      case "--pause-threshold":
        options.pauseThreshold = parseFloat(args[++i]);
        break;
      case "--quiet":
      case "-q":
        options.quiet = true;
        break;
    }
  }

  // Validate model
  if (!MODEL_MAP[options.model]) {
    console.error(`Error: Unknown model "${options.model}". Available: ${Object.keys(MODEL_MAP).join(", ")}`);
    process.exit(1);
  }

  return options;
}

function formatTimestamp(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  const millis = Math.floor((seconds % 1) * 1000);
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")},${String(millis).padStart(3, "0")}`;
}

function detectPauses(words, threshold = 0.2) {
  const pauseIndices = [];
  for (let i = 0; i < words.length - 1; i++) {
    const currentEnd = words[i].timestamp[1] || words[i].timestamp[0];
    const nextStart = words[i + 1].timestamp[0];
    if (nextStart - currentEnd > threshold) {
      pauseIndices.push(i);
    }
  }
  return pauseIndices;
}

function groupByPauses(wordCount, pauseAfterIndices) {
  if (wordCount === 0) return [];
  const groups = [];
  let currentStart = 0;
  
  for (const pauseAfterIdx of pauseAfterIndices) {
    if (pauseAfterIdx >= currentStart && pauseAfterIdx < wordCount) {
      groups.push([currentStart, pauseAfterIdx]);
      currentStart = pauseAfterIdx + 1;
    }
  }
  
  if (currentStart < wordCount) {
    groups.push([currentStart, wordCount - 1]);
  }
  
  return groups;
}

function resizeGroupsToRanges(words, groups, maxSize) {
  const ranges = [];
  
  for (const [groupStart, groupEnd] of groups) {
    const groupWords = words.slice(groupStart, groupEnd + 1);
    const totalLength = groupWords.reduce((acc, w) => acc + w.text.trim().length, 0);
    
    if (totalLength > maxSize && groupWords.length > 1) {
      const relation = totalLength / maxSize;
      const subChunkSize = Math.ceil(groupWords.length / relation);
      const numSubChunks = Math.ceil(groupWords.length / subChunkSize);
      
      for (let i = 0; i < numSubChunks; i++) {
        const localStart = i * subChunkSize;
        const localEnd = Math.min((i + 1) * subChunkSize - 1, groupWords.length - 1);
        if (localStart <= localEnd) {
          ranges.push([groupStart + localStart, groupStart + localEnd]);
        }
      }
    } else {
      ranges.push([groupStart, groupEnd]);
    }
  }
  
  return ranges;
}

function calculateCueRanges(words, pauseAfterIndices, maxSize) {
  const groups = groupByPauses(words.length, pauseAfterIndices);
  return resizeGroupsToRanges(words, groups, maxSize);
}

function buildCueFromRange(words, startIdx, endIdx) {
  const cueWords = words.slice(startIdx, endIdx + 1);
  const text = cueWords.map(w => w.text.trim()).filter(Boolean).join(" ");
  const startTime = cueWords[0].timestamp[0];
  const endTime = cueWords[cueWords.length - 1].timestamp[1] || cueWords[cueWords.length - 1].timestamp[0];
  
  return { text, start: startTime, end: endTime };
}

function buildCuesFromRanges(words, cueRanges) {
  return cueRanges.map(([start, end]) => buildCueFromRange(words, start, end));
}

function generateSRT(cues) {
  const lines = [];
  for (let i = 0; i < cues.length; i++) {
    lines.push(String(i + 1));
    lines.push(`${formatTimestamp(cues[i].start)} --> ${formatTimestamp(cues[i].end)}`);
    lines.push(cues[i].text);
    lines.push("");
  }
  return lines.join("\n");
}

function checkFFmpeg() {
  try {
    execSync("ffmpeg -version", { stdio: "pipe" });
    return true;
  } catch (e) {
    return false;
  }
}

function extractAudio(inputPath, outputPath) {
  // Extract audio to WAV format (16kHz mono, 16-bit) - optimal for Whisper
  const cmd = `ffmpeg -i "${inputPath}" -ar 16000 -ac 1 -c:a pcm_s16le -y "${outputPath}"`;
  execSync(cmd, { stdio: "pipe" });
}

function loadAudioFile(filePath) {
  const buffer = fs.readFileSync(filePath);
  const wav = new WaveFile(buffer);
  
  // Ensure it's 16kHz mono
  if (wav.fmt.sampleRate !== 16000) {
    // Use ffmpeg to resample if needed
    const tempPath = filePath + ".resampled.wav";
    extractAudio(filePath, tempPath);
    const resampledBuffer = fs.readFileSync(tempPath);
    const resampledWav = new WaveFile(resampledBuffer);
    fs.unlinkSync(tempPath);
    return convertToFloat32Array(resampledWav);
  }
  
  return convertToFloat32Array(wav);
}

function convertToFloat32Array(wav) {
  // Get samples based on bit depth
  const bitDepth = wav.fmt.bitsPerSample;
  let samples;
  
  if (bitDepth === 16) {
    samples = wav.getSamples(true, Int16Array);
  } else if (bitDepth === 32) {
    samples = wav.getSamples(true, Int32Array);
  } else {
    // Convert to 16-bit first
    wav.toBitDepth("16");
    samples = wav.getSamples(true, Int16Array);
  }
  
  // Convert to Float32Array normalized to [-1, 1]
  const float32Samples = new Float32Array(samples.length);
  const maxVal = bitDepth === 16 ? 32768 : (bitDepth === 32 ? 2147483648 : 32768);
  
  for (let i = 0; i < samples.length; i++) {
    float32Samples[i] = samples[i] / maxVal;
  }
  
  return float32Samples;
}

async function transcribeFile(options) {
  const { input, model, language, maxChars, pauseThreshold, quiet } = options;
  const modelName = MODEL_MAP[model];
  
  if (!quiet) {
    console.log(`Loading Whisper model: ${model} (${modelName})...`);
    console.log("Note: First run will download the model (may take a few minutes)");
  }
  
  // Create automatic speech recognition pipeline
  // Using ONNX community models compatible with @huggingface/transformers
  const transcriber = await pipeline(
    "automatic-speech-recognition",
    modelName,
    {
      dtype: {
        encoder_model: "fp32",
        decoder_model_merged: "q4",
      },
      device: "cpu",
    }
  );
  
  // Extract audio from video if needed
  const isVideo = /\.(mp4|mov|avi|mkv|webm|flv|wmv|m4v)$/i.test(input);
  let audioPath = input;
  let tempAudioPath = null;
  
  if (isVideo) {
    if (!checkFFmpeg()) {
      throw new Error("ffmpeg is required for video files. Install it first:\n  macOS: brew install ffmpeg\n  Ubuntu: sudo apt-get install ffmpeg");
    }
    
    if (!quiet) {
      console.log("Extracting audio from video...");
    }
    
    tempAudioPath = path.join(os.tmpdir(), `subtitler-${Date.now()}.wav`);
    extractAudio(input, tempAudioPath);
    audioPath = tempAudioPath;
  }
  
  if (!quiet) {
    console.log("Loading audio...");
  }
  
  const audioData = loadAudioFile(audioPath);
  
  if (tempAudioPath) {
    fs.unlinkSync(tempAudioPath);
  }
  
  if (!quiet) {
    console.log(`Transcribing: ${input}...`);
  }
  
  const transcribeOptions = {
    return_timestamps: true,
    chunk_length_s: 30,
    stride_length_s: 5,
  };
  
  if (language) {
    transcribeOptions.language = language;
  }
  
  const result = await transcriber(audioData, transcribeOptions);
  
  // Extract words with timestamps from chunks
  const allWords = [];
  if (result.chunks) {
    for (const chunk of result.chunks) {
      if (chunk.timestamp && chunk.text) {
        allWords.push({
          text: chunk.text.trim(),
          timestamp: chunk.timestamp,
        });
      }
    }
  }
  
  if (!allWords.length && result.text) {
    // Fallback to segment-level timestamps if chunks not available
    const text = result.text.trim();
    if (text) {
      return [{
        text: text,
        start: 0,
        end: audioData.length / 16000, // Estimate duration from sample count
      }];
    }
    return [];
  }
  
  if (!quiet) {
    console.log(`Found ${allWords.length} words with timestamps`);
    console.log(`Grouping into cues (max ${maxChars} chars, pause threshold ${pauseThreshold}s)...`);
  }
  
  const pauseIndices = detectPauses(allWords, pauseThreshold);
  const cueRanges = calculateCueRanges(allWords, pauseIndices, maxChars);
  const cues = buildCuesFromRanges(allWords, cueRanges);
  
  if (!quiet) {
    console.log(`Generated ${cues.length} subtitle cues`);
  }
  
  return cues;
}

async function main() {
  const options = parseArgs();
  
  if (!fs.existsSync(options.input)) {
    console.error(`Error: File not found: ${options.input}`);
    process.exit(1);
  }
  
  const outputPath = options.output || options.input.replace(/\.[^/.]+$/, "") + ".srt";
  
  try {
    const cues = await transcribeFile(options);
    const srtContent = generateSRT(cues);
    fs.writeFileSync(outputPath, srtContent, "utf-8");
    
    if (!options.quiet) {
      console.log("\n" + "=".repeat(50));
      console.log("Transcription complete!");
      console.log(`Model used: ${options.model} (${MODEL_MAP[options.model]})`);
      console.log(`Subtitle cues: ${cues.length}`);
      console.log(`Output saved to: ${path.resolve(outputPath)}`);
      console.log("=".repeat(50));
    }
  } catch (error) {
    console.error(`Error: ${error.message}`);
    process.exit(1);
  }
}

main();
