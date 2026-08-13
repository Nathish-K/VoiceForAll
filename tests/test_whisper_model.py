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
    transcriber = WhisperTranscriber(model_size="base", language="en")
    assert transcriber.model_size == "base"
    assert transcriber.language == "en"
    assert transcriber._model is None
