"""
Unit tests for WhisperTranscriber & WER evaluation (Level 4)
-----------------------------------------------------------
Validates WER metrics, text normalization, and transcriber initialization.
"""

import pytest
from app.speech.whisper_model import WhisperTranscriber, calculate_wer, normalize_text


def test_text_normalization():
    """Verify punctuation stripping and whitespace cleanup."""
    raw = "  Turn ON the Flashlight!  Please...  "
    norm = normalize_text(raw)
    assert norm == "turn on the flashlight please"


def test_wer_calculation_perfect():
    """Verify 0.0 WER on identical strings."""
    ref = "turn on the flashlight"
    hyp = "Turn on the flashlight!"
    assert calculate_wer(ref, hyp) == 0.0


def test_wer_calculation_errors():
    """Verify WER calculation with deletions, substitutions, and insertions."""
    ref = "turn on the flashlight"
    
    # 1 deletion: "the" is missing -> 1 error out of 4 words = 0.25 (25%)
    hyp_del = "turn on flashlight"
    assert calculate_wer(ref, hyp_del) == 0.25

    # 1 substitution: "torch" instead of "flashlight" -> 1 / 4 = 0.25
    hyp_sub = "turn on the torch"
    assert calculate_wer(ref, hyp_sub) == 0.25

    # 1 insertion: "please" added -> 1 / 4 = 0.25
    hyp_ins = "please turn on the flashlight"
    assert calculate_wer(ref, hyp_ins) == 0.25


def test_whisper_transcriber_initialization():
    """Verify transcriber initial state without loading weights."""
    transcriber = WhisperTranscriber(model_size="base.en", language="en")
    assert transcriber.model_size == "base.en"
    assert transcriber.language == "en"
    assert transcriber._model is None


def test_model_size_switching():
    """Verify switching between valid Whisper tiers."""
    transcriber = WhisperTranscriber(model_size="base")
    transcriber.set_model_size("base.en")
    assert transcriber.model_size == "base.en"
    transcriber.set_model_size("tiny.en")
    assert transcriber.model_size == "tiny.en"

    with pytest.raises(ValueError):
        transcriber.set_model_size("invalid_tier")


def test_trim_silence_vad():
    """Verify VAD trims leading/trailing dead silence while preserving speech segment."""
    import numpy as np
    from app.speech.whisper_model import trim_silence_vad

    sample_rate = 16000
    # 1 second of silence, 1 second of active sine wave, 1 second of silence
    t_silence = np.zeros(sample_rate, dtype=np.float32)
    t_active = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate)).astype(np.float32) * 0.8
    full_audio = np.concatenate([t_silence, t_active, t_silence])

    trimmed = trim_silence_vad(full_audio, sample_rate=sample_rate)
    # Trimmed length should be significantly shorter than the original 3 seconds (48000 samples)
    assert len(trimmed) < len(full_audio)
    # But should still contain the active speech
    assert np.max(np.abs(trimmed)) > 0.5

