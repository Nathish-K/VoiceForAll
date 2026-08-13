# 🎙️ Voice Assistant - Level 1 to Level 4 Pipeline

End-to-end local Python pipeline for recording, signal inspection, noise reduction, and automatic speech recognition (ASR) with OpenAI Whisper.

```
🎙️ Microphone ──► 💾 Raw WAV ──► 📊 Analysis ──► 🔇 Noise Reduction ──► 📁 Clean WAV ──► 🗣️ Whisper ──► 📝 Text
```

---

## 📁 Project Structure

```
VoiceForAll/
│
├── app/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── recorder.py          # Level 1: Microphone capture (16 kHz Mono WAV)
│   │   ├── analyzer.py          # Level 2: RMS calculation, Waveforms & Spectrograms
│   │   └── noise_reducer.py     # Level 3: Spectral Gating Noise Reduction & SNR
│   │
│   ├── speech/
│   │   ├── __init__.py
│   │   └── whisper_model.py     # Level 4: Whisper ASR, WER Evaluation & Comparison
│   │
│   └── main.py                  # Unified interactive CLI (Record, Clean, Transcribe)
│
├── audio_samples/
│   ├── raw/                     # Original recorded WAV files
│   ├── cleaned/                 # De-noised WAV files
│   └── analysis/                # Visual waveform & spectrogram comparison artifacts
│
├── tests/
│   ├── test_recorder.py         # Level 1 unit tests
│   ├── test_analyzer.py         # Level 2 unit tests
│   ├── test_noise_reducer.py    # Level 3 unit tests
│   └── test_whisper_model.py    # Level 4 unit tests
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Activate Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Run All Tests (Levels 1, 2, 3, 4)
```powershell
pytest -v tests/
```

### 3. Run the Interactive CLI
```powershell
python app/main.py
```

---

## 📊 Word Error Rate (WER) Formula

$$\mathbf{\text{WER} = \frac{\text{Substitutions } (S) + \text{Deletions } (D) + \text{Insertions } (I)}{\text{Total Reference Words } (N)}}$$

* $\text{WER} = 0.0$ ($0\%$): Perfect transcription.
* $\Delta \text{WER} = \text{Raw WER} - \text{Cleaned WER}$: Measures accuracy improvement achieved by Level 3 noise reduction.
