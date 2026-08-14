"""
Whisper Speech Recognition Module (Level 4)
------------------------------------------
Responsible for loading OpenAI's Whisper model, transcribing 16 kHz WAV audio
into text strings, saving transcriptions to `transcriptions/`, computing
Word Error Rate (WER), and comparing raw vs cleaned audio accuracy.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Union
import numpy as np
import scipy.io.wavfile as wavfile


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRANSCRIPTIONS_DIR = PROJECT_ROOT / "transcriptions"
TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TranscriptionResult:
    """Structured result of a single Whisper audio transcription."""
    text: str
    language: str
    inference_time_sec: float
    model_size: str
    audio_file_path: Optional[Path] = None
    saved_text_path: Optional[Path] = None


@dataclass
class ComparisonReport:
    """Comparison report between raw noisy audio and cleaned audio transcriptions."""
    raw_audio_path: Path
    cleaned_audio_path: Path
    model_size: str
    raw_text: str
    cleaned_text: str
    reference_text: Optional[str]
    raw_wer: Optional[float]
    cleaned_wer: Optional[float]
    wer_improvement: Optional[float]
    conclusion: str


def normalize_text(text: str) -> str:
    """Cleans punctuation and extra whitespace for standardized text evaluation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculates Word Error Rate (WER) between a reference string and model hypothesis.

    Formula:
        WER = (Substitutions + Deletions + Insertions) / Total Reference Words

    Args:
        reference (str): Ground-truth expected phrase (e.g. "turn on the flashlight").
        hypothesis (str): Whisper's transcribed output.

    Returns:
        float: WER value (0.0 = perfect match, 1.0 = 100% error rate).
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm and not hyp_norm:
        return 0.0
    if not ref_norm:
        return 1.0

    try:
        import jiwer
        return float(jiwer.wer(ref_norm, hyp_norm))
    except ImportError:
        # Fallback Levenshtein dynamic programming distance calculation
        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()

        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
        for i in range(len(ref_words) + 1):
            d[i, 0] = i
        for j in range(len(hyp_words) + 1):
            d[0, j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i, j] = d[i - 1, j - 1]
                else:
                    substitution = d[i - 1, j - 1] + 1
                    insertion = d[i, j - 1] + 1
                    deletion = d[i - 1, j] + 1
                    d[i, j] = min(substitution, insertion, deletion)

        return float(d[len(ref_words), len(hyp_words)] / len(ref_words))


def trim_silence_vad(
    audio_float: np.ndarray,
    sample_rate: int = 16000,
    frame_duration_ms: float = 30.0,
    energy_threshold_ratio: float = 0.03,
    padding_ms: float = 200.0,
) -> np.ndarray:
    """
    Trims leading and trailing silence from speech audio using RMS energy thresholding,
    with a padding margin to protect onset plosives and offset consonants.

    Args:
        audio_float (np.ndarray): 1D float32 audio array.
        sample_rate (int): Sampling rate in Hz.
        frame_duration_ms (float): Frame duration in ms (default: 30ms).
        energy_threshold_ratio (float): Fraction of peak energy considered active speech.
        padding_ms (float): Padding in ms around active speech boundaries.

    Returns:
        np.ndarray: Trimmed float32 audio array.
    """
    if len(audio_float) < sample_rate * 0.4:
        return audio_float

    frame_size = int((frame_duration_ms / 1000.0) * sample_rate)
    if frame_size <= 0:
        return audio_float

    num_frames = len(audio_float) // frame_size
    if num_frames < 2:
        return audio_float

    frames = audio_float[: num_frames * frame_size].reshape(num_frames, frame_size)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))

    peak_rms = np.max(frame_rms)
    if peak_rms < 1e-4:
        return audio_float  # Entirely silent

    threshold = max(0.003, peak_rms * energy_threshold_ratio)
    active_indices = np.where(frame_rms >= threshold)[0]

    if len(active_indices) == 0:
        return audio_float

    start_frame = active_indices[0]
    end_frame = active_indices[-1]

    padding_frames = int((padding_ms / 1000.0) * (sample_rate / frame_size))
    start_sample = max(0, (start_frame - padding_frames) * frame_size)
    end_sample = min(len(audio_float), (end_frame + 1 + padding_frames) * frame_size)

    return audio_float[start_sample:end_sample]


class WhisperTranscriber:
    """
    Whisper ASR Transcriber.

    Attributes:
        model_size (str): Whisper model tier ('tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium').
            Default is 'base' (~74M parameters, fast CPU inference).
        device (str): Inference device ('cpu' or 'cuda').
        language (str): Target spoken language code (e.g. 'en' for English).
    """

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        language: str = "en",
    ) -> None:
        self.model_size = model_size
        self.language = language
        self._model = None

        if device:
            self.device = device
        else:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

    def set_model_size(self, model_size: str) -> None:
        """Dynamically switches Whisper model tier ('tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium')."""
        valid_tiers = ("tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en")
        if model_size not in valid_tiers:
            raise ValueError(f"Invalid model size '{model_size}'. Choose from {valid_tiers}")
        if model_size != self.model_size:
            self.model_size = model_size
            self._model = None

    def _ensure_model_loaded(self) -> None:
        """Loads the Whisper model into memory on first call (lazy-loading)."""
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.model_size, device=self.device)

    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        save_txt: bool = True,
        vad_trim: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribes a WAV file or NumPy audio array into text with high accuracy.

        Args:
            audio_input (str | Path | np.ndarray): File path or 16 kHz float32/int16 array.
            save_txt (bool): If True and audio_input is a file, saves text to transcriptions/<name>.txt.
            vad_trim (bool): If True, applies Voice Activity Detection trimming to silence margins.

        Returns:
            TranscriptionResult: Recognized text and metadata.
        """
        self._ensure_model_loaded()

        file_path_obj = None
        saved_txt_path = None
        start_time = time.time()

        # Handle file path vs NumPy array input
        if isinstance(audio_input, (str, Path)):
            file_path_obj = Path(audio_input).resolve()
            if not file_path_obj.is_file():
                raise FileNotFoundError(f"Audio file not found: {file_path_obj}")

            sr, raw_data = wavfile.read(str(file_path_obj))
            flat_audio = raw_data.flatten()
            if flat_audio.dtype == np.int16:
                float_audio = flat_audio.astype(np.float32) / 32768.0
            else:
                float_audio = flat_audio.astype(np.float32)

            if sr != 16000:
                import math
                from scipy import signal
                g = math.gcd(int(16000), int(sr))
                up = int(16000 // g)
                down = int(sr // g)
                float_audio = signal.resample_poly(float_audio, up, down)

        elif isinstance(audio_input, np.ndarray):
            flat_audio = audio_input.flatten()
            if flat_audio.dtype == np.int16:
                float_audio = flat_audio.astype(np.float32) / 32768.0
            else:
                float_audio = flat_audio.astype(np.float32)
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        # Audio peak normalization prior to Whisper inference
        peak = np.max(np.abs(float_audio))
        if peak > 1e-4 and peak < 0.90:
            float_audio = (float_audio / peak) * 0.95

        # Voice Activity Detection (VAD) silence trimming
        if vad_trim:
            float_audio = trim_silence_vad(float_audio, sample_rate=16000)

        # Transcribe parameters optimized for maximum accuracy
        transcribe_kwargs = {
            "temperature": 0.0,
            "beam_size": 5,
            "best_of": 5,
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.6,
            "compression_ratio_threshold": 2.4,
            "fp16": (self.device == "cuda"),
        }

        # For multilingual models, pass language specification
        if not self.model_size.endswith(".en") and self.language:
            transcribe_kwargs["language"] = self.language

        # Execute local Whisper inference
        result = self._model.transcribe(
            float_audio,
            **transcribe_kwargs,
        )

        inference_time = round(time.time() - start_time, 2)
        recognized_text = result.get("text", "").strip()
        detected_lang = result.get("language", self.language)

        # Save transcription to transcriptions/ directory
        if save_txt and file_path_obj:
            saved_txt_path = TRANSCRIPTIONS_DIR / f"{file_path_obj.stem}.txt"
            saved_txt_path.write_text(recognized_text, encoding="utf-8")

        return TranscriptionResult(
            text=recognized_text,
            language=detected_lang,
            inference_time_sec=inference_time,
            model_size=self.model_size,
            audio_file_path=file_path_obj,
            saved_text_path=saved_txt_path,
        )

    def batch_transcribe(self, file_paths: List[Union[str, Path]]) -> List[TranscriptionResult]:
        """Transcribes multiple audio files in sequence."""
        return [self.transcribe(fp) for fp in file_paths]

    def compare_raw_vs_cleaned(
        self,
        raw_audio_path: Union[str, Path],
        cleaned_audio_path: Union[str, Path],
        reference_text: Optional[str] = None,
    ) -> ComparisonReport:
        """
        Transcribes both raw and noise-reduced versions of a recording and evaluates
        recognition results and optional WER improvement.

        Args:
            raw_audio_path (str | Path): Path to raw noisy WAV.
            cleaned_audio_path (str | Path): Path to cleaned WAV.
            reference_text (str, optional): Ground-truth phrase spoken.

        Returns:
            ComparisonReport: Comparative transcription and WER metrics.
        """
        raw_path = Path(raw_audio_path).resolve()
        clean_path = Path(cleaned_audio_path).resolve()

        raw_res = self.transcribe(raw_path)
        clean_res = self.transcribe(clean_path)

        raw_wer = None
        clean_wer = None
        wer_diff = None
        conclusion = "Transcription comparison complete."

        if reference_text:
            raw_wer = round(calculate_wer(reference_text, raw_res.text), 4)
            clean_wer = round(calculate_wer(reference_text, clean_res.text), 4)
            wer_diff = round(raw_wer - clean_wer, 4)

            if wer_diff > 0:
                conclusion = f"Noise reduction improved transcription accuracy by {wer_diff*100:.1f}% WER reduction."
            elif wer_diff == 0 and clean_wer == 0:
                conclusion = "Perfect transcription achieved on both raw and cleaned audio (0.0% WER)."
            elif wer_diff == 0:
                conclusion = "Both raw and cleaned audio produced identical recognition accuracy."
            else:
                conclusion = f"Cleaned WER was higher by {abs(wer_diff)*100:.1f}%. Possible slight speech attenuation."
        else:
            if raw_res.text == clean_res.text:
                conclusion = "Identical transcription produced for both raw and cleaned audio."
            else:
                conclusion = "Cleaned audio produced a different transcription output from raw audio."

        return ComparisonReport(
            raw_audio_path=raw_path,
            cleaned_audio_path=clean_path,
            model_size=self.model_size,
            raw_text=raw_res.text,
            cleaned_text=clean_res.text,
            reference_text=reference_text,
            raw_wer=raw_wer,
            cleaned_wer=clean_wer,
            wer_improvement=wer_diff,
            conclusion=conclusion,
        )


def transcribe_audio(
    audio_path: Union[str, Path],
    model_size: str = "base",
    save_txt: bool = True,
) -> str:
    """
    Convenience function: receives an audio file path, loads Whisper,
    transcribes the audio, saves the text, and returns the transcription string.
    """
    transcriber = WhisperTranscriber(model_size=model_size)
    result = transcriber.transcribe(audio_path, save_txt=save_txt)
    return result.text
