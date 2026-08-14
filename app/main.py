"""
Voice Assistant - Level 1, 2, 3 & 4: Record, Analyze, Clean & Transcribe
-------------------------------------------------------------------------
Unified CLI pipeline for:
1. Capturing 16 kHz Mono audio (Level 1).
2. Audio Analysis & Statistics (Level 2).
3. Spectral Gating Noise Reduction (Level 3).
4. OpenAI Whisper Speech-to-Text & WER Evaluation (Level 4).
"""

from datetime import datetime
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.audio.recorder import AudioRecorder  # noqa: E402
from app.audio.analyzer import AudioAnalyzer  # noqa: E402
from app.audio.noise_reducer import NoiseReducer, NoiseReductionReport  # noqa: E402
from app.speech.whisper_model import WhisperTranscriber, calculate_wer  # noqa: E402


def print_banner() -> None:
    """Prints application banner."""
    print("=" * 72)
    print(" 🎙️  VOICE ASSISTANT - LEVEL 1 TO 4 PIPELINE")
    print("    Record → Analyze (RMS/Spectrogram) → Clean (Noise) → Transcribe (Whisper)")
    print("=" * 72)


def get_user_duration() -> float:
    """Prompts user for duration with validation."""
    default_duration = 5.0
    while True:
        user_input = input(f"\nEnter recording duration in seconds [default: {default_duration}s]: ").strip()
        if not user_input:
            return default_duration
        try:
            duration = float(user_input)
            if duration <= 0:
                print("⚠️  Duration must be greater than 0. Please try again.")
                continue
            if duration > 300:
                print("⚠️  Duration must be 300 seconds (5 minutes) or less.")
                continue
            return duration
        except ValueError:
            print("⚠️  Invalid input! Please enter a valid number (e.g. 3, 5, 10.5).")


def get_noise_tag() -> str:
    """Prompts user for a noise category label."""
    print("\nSelect the environmental noise condition:")
    print(" 1. 🌀 Fan / AC Noise (Stationary)")
    print(" 2. ⌨️  Keyboard Typing Noise (Impulsive Non-Stationary)")
    print(" 3. 👥 Background Conversation (Babble Noise)")
    print(" 4. 🚗 Traffic / Outdoor Noise")
    print(" 5. 🏠 General Room Noise")
    choice = input("Enter option (1-5) [default: 1]: ").strip()

    tags = {
        "1": "fan_noise",
        "2": "keyboard_noise",
        "3": "babble_noise",
        "4": "traffic_noise",
        "5": "room_noise",
    }
    return tags.get(choice, "fan_noise")


def list_recordings(subdir: str = "raw") -> list[Path]:
    """Returns sorted list of audio files from raw or cleaned folder."""
    target_dir = PROJECT_ROOT / "audio_samples" / subdir
    if not target_dir.exists():
        return []
    return sorted(list(target_dir.glob("*.wav")), key=lambda p: p.stat().st_mtime, reverse=True)


def get_model_size_choice() -> str:
    """Prompts user to select Whisper model tier."""
    print("\nSelect Whisper Model Tier:")
    print(" 1. ⚡ Tiny  (~39M params, fastest)")
    print(" 2. 🎯 Base  (~74M params, recommended default)")
    print(" 3. 🧠 Small (~244M params, higher accuracy)")
    choice = input("Enter option (1-3) [default: 2]: ").strip()
    models = {"1": "tiny", "2": "base", "3": "small"}
    return models.get(choice, "base")


def display_level4_report(
    raw_path: Path,
    clean_path: Path,
    raw_text: str,
    clean_text: str,
    raw_time: float,
    clean_time: float,
    model_size: str = "base",
    reference_text: str | None = None,
) -> None:
    """Displays the comprehensive Level 4 speech recognition comparative report."""
    print("\n" + "=" * 72)
    print(" 🗣️  LEVEL 4: WHISPER SPEECH RECOGNITION REPORT")
    print("=" * 72)
    print(f" • Raw Audio File       : {raw_path.name}")
    print(f" • Cleaned Audio File   : {clean_path.name}")
    print(f" • Model Tier           : Whisper '{model_size}' (Local inference)")
    print("-" * 72)
    print(" 📝 TRANSCRIPTION RESULTS")
    print("-" * 72)
    print(f" 🔴 RAW AUDIO Output   : \"{raw_text}\" (Inference: {raw_time:.2f}s)")
    print(f" 🟢 CLEANED AUDIO Output: \"{clean_text}\" (Inference: {clean_time:.2f}s)")

    if reference_text:
        raw_wer = calculate_wer(reference_text, raw_text)
        clean_wer = calculate_wer(reference_text, clean_text)
        diff_wer = raw_wer - clean_wer

        print("-" * 72)
        print(" 🎯 WORD ERROR RATE (WER) EVALUATION")
        print("-" * 72)
        print(f" • Reference (Expected) : \"{reference_text}\"")
        print(f" • Raw Audio WER        : {raw_wer*100:.1f}%")
        print(f" • Cleaned Audio WER    : {clean_wer*100:.1f}%")
        if diff_wer > 0:
            print(f" • Accuracy Improvement : ▲ +{diff_wer*100:.1f}% WER reduction from noise removal!")
        elif diff_wer == 0:
            print(" • Accuracy Comparison  : ⚖️ Both raw and cleaned audio produced identical WER.")
        else:
            print(f" • Accuracy Comparison  : ⚠️ Cleaned WER higher by {abs(diff_wer)*100:.1f}%. Possible slight speech attenuation.")

    print("=" * 72 + "\n")


def main() -> None:
    """Main CLI execution flow."""
    print_banner()

    model_size = get_model_size_choice()
    recorder = AudioRecorder(sample_rate=16000, channels=1, dtype="int16")
    analyzer = AudioAnalyzer()
    reducer = NoiseReducer(prop_decrease=0.20, noise_profile_duration_sec=1.0)
    transcriber = WhisperTranscriber(model_size=model_size)

    raw_files = list_recordings("raw")
    cleaned_files = list_recordings("cleaned")

    print("\nSelect an action:")
    print(" 1. 🎙️  Record NEW Audio → Clean → Transcribe with Whisper (Full Pipeline)")
    if cleaned_files:
        print(f" 2. 🗣️  Transcribe MOST RECENT Cleaned Audio ({cleaned_files[0].name})")
    if raw_files:
        print(f" 3. 🔬 Compare Raw vs Cleaned Transcription on Most Recent ({raw_files[0].name})")
    choice = input("\nEnter choice [default: 1]: ").strip()

    if choice == "2" and cleaned_files:
        target_file = cleaned_files[0]
        print(f"\nTranscribing: {target_file.name} with Whisper '{model_size}'...")
        res = transcriber.transcribe(target_file)
        print("\n" + "=" * 60)
        print(f"📁 File          : {target_file.name}")
        print(f"🗣️ Transcription : \"{res.text}\"")
        print(f"⏱️ Inference Time: {res.inference_time_sec:.2f} seconds")
        print("=" * 60 + "\n")
        return

    if choice == "3" and raw_files:
        raw_target = raw_files[0]
        cleaned_target = PROJECT_ROOT / "audio_samples" / "cleaned" / f"{raw_target.stem}_cleaned.wav"
        if not cleaned_target.exists():
            print(f"Generating cleaned version for {raw_target.name}...")
            report = reducer.clean_audio_file(raw_target, cleaned_target)
            if report.speech_attenuation_warning:
                print(f"⚠️  {report.speech_attenuation_warning}")

        ref_input = input("\nEnter expected reference text (leave blank to skip WER): ").strip()
        ref_text = ref_input if ref_input else None

        print(f"\nTranscribing raw audio with Whisper '{model_size}'...")
        raw_res = transcriber.transcribe(raw_target)
        print(f"Transcribing cleaned audio with Whisper '{model_size}'...")
        clean_res = transcriber.transcribe(cleaned_target)

        display_level4_report(
            raw_target,
            cleaned_target,
            raw_res.text,
            clean_res.text,
            raw_res.inference_time_sec,
            clean_res.inference_time_sec,
            model_size=model_size,
            reference_text=ref_text,
        )
        return

    # Action 1: Record, Clean, Transcribe
    try:
        device_info = recorder.check_input_device()
        print(f"\n[Hardware] Default Microphone: '{device_info.get('name', 'Unknown')}'")
    except RuntimeError as exc:
        print(f"\n❌ Microphone Error: {exc}")
        sys.exit(1)

    noise_tag = get_noise_tag()
    duration = get_user_duration()
    ref_input = input("\nEnter expected phrase (for WER tracking after transcription) [optional]: ").strip()
    reference_text = ref_input if ref_input else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = PROJECT_ROOT / "audio_samples" / "raw" / f"{noise_tag}_{timestamp}.wav"
    clean_path = PROJECT_ROOT / "audio_samples" / "cleaned" / f"{noise_tag}_{timestamp}_cleaned.wav"

    for i in range(3, 0, -1):
        print(f"Starting in {i}...", end="\r", flush=True)
        time.sleep(1)
    print(f"🔴 RECORDING NOW for {duration} seconds... Speak clearly into your microphone!")

    try:
        # 1. Record with peak auto-normalization
        audio_data = recorder.record(duration=duration, auto_normalize=True)
        saved_raw = recorder.save_wav(file_path=raw_path, audio_data=audio_data)
        print(f"💾 Saved normalized raw audio to: {saved_raw.name}")

        # 2. Clean
        print("🔇 Applying Spectral Gating Noise Reduction...")
        reduction_report = reducer.clean_audio_file(saved_raw, clean_path)
        print(f"💾 Saved cleaned audio to: {clean_path.name}")
        if reduction_report.speech_attenuation_warning:
            print(f"⚠️  {reduction_report.speech_attenuation_warning}")

        # 3. Transcribe Raw & Cleaned with Whisper
        print(f"🗣️ Transcribing Raw Audio with Whisper '{model_size}'...")
        raw_res = transcriber.transcribe(saved_raw)

        print(f"🗣️ Transcribing Cleaned Audio with Whisper '{model_size}'...")
        clean_res = transcriber.transcribe(clean_path)

        # 4. Generate comparison plots
        analyzer.plot_comparison(
            saved_raw,
            clean_path,
            filename=f"{saved_raw.stem}_comparison.png",
        )

        # 5. Display Full Report
        display_level4_report(
            saved_raw,
            clean_path,
            raw_res.text,
            clean_res.text,
            raw_res.inference_time_sec,
            clean_res.inference_time_sec,
            model_size=model_size,
            reference_text=reference_text,
        )

    except KeyboardInterrupt:
        print("\n\n⚠️ Process cancelled by user (Ctrl+C).")
        sys.exit(0)
    except Exception as exc:
        print(f"\n❌ Error during processing: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

