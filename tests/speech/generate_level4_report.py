"""
Level 4 Final Report Generator & Noise Condition Experiment
------------------------------------------------------------
Compares Raw vs Cleaned audio recognition accuracy, calculates WER,
and outputs a structured ML report.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.speech.whisper_model import WhisperTranscriber, calculate_wer


def generate_report(
    raw_audio_path: Path,
    cleaned_audio_path: Path,
    reference_text: str | None = None,
    model_size: str = "base",
) -> None:
    transcriber = WhisperTranscriber(model_size=model_size)

    raw_res = transcriber.transcribe(raw_audio_path, save_txt=True)
    clean_res = transcriber.transcribe(cleaned_audio_path, save_txt=True)

    print("\n" + "=" * 62)
    print(" LEVEL 4 — SPEECH-TO-TEXT WITH WHISPER")
    print("=" * 62)
    print(f"\nWhisper Model:\n{model_size}")
    print(f"\nRaw Audio File:\n{raw_audio_path.name}")
    print(f"\nCleaned Audio File:\n{cleaned_audio_path.name}")
    print(f"\nRaw Transcription:\n{raw_res.text if raw_res.text else '<Silence / No speech detected>'}")
    print(f"\nCleaned Transcription:\n{clean_res.text if clean_res.text else '<Silence / No speech detected>'}")

    if reference_text:
        raw_wer = calculate_wer(reference_text, raw_res.text)
        clean_wer = calculate_wer(reference_text, clean_res.text)
        wer_diff = raw_wer - clean_wer

        print(f"\nReference Text:\n{reference_text}")
        print(f"\nRaw WER:\n{raw_wer*100:.1f}%")
        print(f"\nCleaned WER:\n{clean_wer*100:.1f}%")
        print(f"\nWER Improvement:\n{'+' if wer_diff > 0 else ''}{wer_diff*100:.1f}%")

        if wer_diff > 0:
            conclusion = f"Noise reduction improved ASR accuracy by {wer_diff*100:.1f}% WER reduction."
        elif wer_diff == 0 and clean_wer == 0:
            conclusion = "Perfect transcription achieved on both raw and cleaned audio (0.0% WER)."
        elif wer_diff == 0:
            conclusion = "Both raw and cleaned audio produced identical recognition accuracy."
        else:
            conclusion = f"Cleaned WER was higher by {abs(wer_diff)*100:.1f}%. Possible slight speech attenuation."

        print(f"\nConclusion:\n{conclusion}")
    else:
        if raw_res.text == clean_res.text:
            print("\nConclusion:\nBoth raw and cleaned audio produced identical transcription.")
        else:
            print("\nConclusion:\nCleaned audio produced a different transcription from raw audio.")

    print(f"\nSaved Transcriptions:")
    print(f" • Raw     : transcriptions/{raw_audio_path.stem}.txt")
    print(f" • Cleaned : transcriptions/{cleaned_audio_path.stem}.txt")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    # Test on existing recording
    raw_dir = PROJECT_ROOT / "audio_samples" / "raw"
    clean_dir = PROJECT_ROOT / "audio_samples" / "cleaned"

    raw_file = raw_dir / "fan_noise_20260813_232257.wav"
    clean_file = clean_dir / "fan_noise_20260813_232257_cleaned.wav"

    if raw_file.exists() and clean_file.exists():
        generate_report(raw_file, clean_file, reference_text=None)
    else:
        print("Required test files not found. Please ensure recordings exist in audio_samples/raw and audio_samples/cleaned.")
