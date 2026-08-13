"""
Test script for batch processing multiple audio command files with Whisper.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.speech.whisper_model import WhisperTranscriber, calculate_wer


def run_batch_test() -> None:
    transcriber = WhisperTranscriber(model_size="base")

    # Look for files in tests/speech/ or audio_samples/cleaned/
    test_dir = PROJECT_ROOT / "tests" / "speech"
    cleaned_dir = PROJECT_ROOT / "audio_samples" / "cleaned"

    wav_files = list(test_dir.glob("*.wav")) + list(cleaned_dir.glob("*.wav"))

    print("=" * 65)
    print(" 🗣️  WHISPER MULTI-FILE TEST RESULTS")
    print("=" * 65)

    if not wav_files:
        print("No WAV files found for testing yet.")
        print(f"Place test WAV files in: {test_dir}")
        print("=" * 65)
        return

    print(f"{'Audio File':<35} {'Whisper Output'}")
    print("-" * 65)

    for audio_file in wav_files:
        result = transcriber.transcribe(audio_file)
        print(f"{audio_file.name:<35} {result.text if result.text else '<Silence / No speech detected>'}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_batch_test()
