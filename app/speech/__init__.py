"""Speech recognition (ASR) package powered by Whisper."""

from app.speech.whisper_model import (
    WhisperTranscriber,
    TranscriptionResult,
    ComparisonReport,
    transcribe_audio,
    calculate_wer,
)

__all__ = [
    "WhisperTranscriber",
    "TranscriptionResult",
    "ComparisonReport",
    "transcribe_audio",
    "calculate_wer",
]
