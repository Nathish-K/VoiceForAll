"""
Noise Reducer Module (Level 3)
-----------------------------
Responsible for spectral gating noise reduction on raw audio recordings.
Uses `noisereduce` (STFT spectral masking), preserves original audio files,
and calculates Signal-to-Noise Ratio (SNR) improvements.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import noisereduce as nr
import numpy as np
import scipy.io.wavfile as wavfile


@dataclass
class NoiseReductionReport:
    """Structured report comparing raw audio vs cleaned audio metrics."""
    raw_file_path: Path
    cleaned_file_path: Path
    sample_rate: int
    duration_seconds: float
    raw_rms: float
    cleaned_rms: float
    rms_reduction_pct: float
    raw_peak: int
    cleaned_peak: int
    raw_snr_db: float
    cleaned_snr_db: float
    snr_improvement_db: float


class NoiseReducer:
    """
    Modular Audio Noise Reducer using Spectral Gating.

    Attributes:
        prop_decrease (float): Proportion of estimated noise to reduce (0.0 to 1.0).
            Default is 0.80 (80%). Choosing 80-85% prevents robotic phase artifacts.
        stationary (bool): True for steady stationary noise (fans, AC),
            False for dynamic non-stationary background noise.
        n_fft (int): Number of FFT points for frequency decomposition (default: 1024).
        win_length (int): Length of each analysis window (default: 512).
        hop_length (int): Step size between successive windows (default: 128).
    """

    def __init__(
        self,
        prop_decrease: float = 0.80,
        stationary: bool = True,
        n_fft: int = 1024,
        win_length: int = 512,
        hop_length: int = 128,
    ) -> None:
        if not (0.0 <= prop_decrease <= 1.0):
            raise ValueError(f"prop_decrease must be between 0.0 and 1.0, got {prop_decrease}")

        self.prop_decrease = prop_decrease
        self.stationary = stationary
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

    @staticmethod
    def estimate_snr(audio_data: np.ndarray, top_percentile: float = 90, bottom_percentile: float = 15) -> float:
        """
        Estimates Signal-to-Noise Ratio (SNR) in Decibels (dB) from a single recording
        by comparing active high-energy speech frames against lowest-energy background floor frames.

        Formula:
            SNR (dB) = 10 * log10( Power_speech / Power_noise )

        Args:
            audio_data (np.ndarray): Audio samples.
            top_percentile (float): Percentile threshold for active speech energy.
            bottom_percentile (float): Percentile threshold for noise floor energy.

        Returns:
            float: Estimated SNR in Decibels (dB). Higher is cleaner.
        """
        samples = audio_data.flatten().astype(np.float64)
        if len(samples) < 100 or np.all(samples == 0):
            return 0.0

        # Frame-by-frame energy (frame size: ~20ms = 320 samples at 16kHz)
        frame_size = 320
        num_frames = len(samples) // frame_size
        if num_frames == 0:
            return 0.0

        frames = samples[: num_frames * frame_size].reshape(num_frames, frame_size)
        frame_powers = np.mean(np.square(frames), axis=1)

        # Estimate signal power (upper percentile) and noise power (lower percentile)
        signal_power = np.percentile(frame_powers, top_percentile)
        noise_power = np.percentile(frame_powers, bottom_percentile)

        # Floor noise power to avoid division by zero or log(0)
        noise_power = max(noise_power, 1e-10)
        signal_power = max(signal_power, 1e-10)

        snr_linear = signal_power / noise_power
        snr_db = float(10.0 * np.log10(snr_linear))
        return round(snr_db, 2)

    def clean_audio_array(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        noise_clip: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Applies spectral gating noise reduction on a NumPy audio array.

        Args:
            audio_data (np.ndarray): Input audio array (int16 or float32).
            sample_rate (int): Sampling rate in Hz (default: 16000).
            noise_clip (np.ndarray, optional): Standalone silence/noise segment to learn profile.

        Returns:
            np.ndarray: Cleaned audio array matching the input dtype.
        """
        original_dtype = audio_data.dtype
        # Convert int16 to float32 normalized [-1.0, 1.0] for high-precision math
        if original_dtype == np.int16:
            norm_audio = audio_data.astype(np.float32) / 32768.0
            norm_noise = (noise_clip.astype(np.float32) / 32768.0) if noise_clip is not None else None
        else:
            norm_audio = audio_data.astype(np.float32)
            norm_noise = noise_clip.astype(np.float32) if noise_clip is not None else None

        # Apply noisereduce spectral gating
        cleaned_float = nr.reduce_noise(
            y=norm_audio.flatten(),
            sr=sample_rate,
            y_noise=norm_noise.flatten() if norm_noise is not None else None,
            prop_decrease=self.prop_decrease,
            stationary=self.stationary,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            time_mask_smooth_ms=64,
        )

        # Reshape back to match original channels (e.g. (N, 1) or (N,))
        if audio_data.ndim > 1:
            cleaned_float = cleaned_float.reshape(-1, audio_data.shape[1])

        # Convert back to original dtype
        if original_dtype == np.int16:
            # Clip between -1.0 and 1.0 before int16 scaling to prevent integer overflow
            clipped = np.clip(cleaned_float, -1.0, 1.0)
            return (clipped * 32767.0).astype(np.int16)
        else:
            return cleaned_float.astype(original_dtype)

    def clean_audio_file(
        self,
        raw_file_path: Union[str, Path],
        output_file_path: Optional[Union[str, Path]] = None,
        noise_profile_duration_sec: float = 0.5,
    ) -> NoiseReductionReport:
        """
        Reads a raw WAV file, applies spectral gating noise reduction, saves the cleaned
        WAV file into `audio_samples/cleaned/`, and computes comparative metrics.

        Args:
            raw_file_path (str | Path): Source noisy WAV file.
            output_file_path (str | Path, optional): Output path. Defaults to audio_samples/cleaned/<name>_cleaned.wav.
            noise_profile_duration_sec (float): Duration of initial silence to use as noise profile.

        Returns:
            NoiseReductionReport: Comprehensive before/after report.
        """
        raw_path = Path(raw_file_path).resolve()
        if not raw_path.is_file():
            raise FileNotFoundError(f"Input file not found: {raw_path}")

        # Determine target cleaned path
        if output_file_path:
            clean_path = Path(output_file_path).resolve()
        else:
            project_root = Path(__file__).resolve().parent.parent.parent
            cleaned_dir = project_root / "audio_samples" / "cleaned"
            cleaned_dir.mkdir(parents=True, exist_ok=True)
            clean_path = cleaned_dir / f"{raw_path.stem}_cleaned.wav"

        clean_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Load raw audio
        sample_rate, raw_data = wavfile.read(str(raw_path))

        # Extract first N seconds as explicit noise clip if available
        noise_samples_count = int(noise_profile_duration_sec * sample_rate)
        noise_clip = raw_data[:noise_samples_count] if len(raw_data) > noise_samples_count else None

        # 2. Perform noise reduction
        cleaned_data = self.clean_audio_array(
            audio_data=raw_data,
            sample_rate=sample_rate,
            noise_clip=noise_clip,
        )

        # 3. Save cleaned WAV file
        wavfile.write(str(clean_path), sample_rate, cleaned_data)

        # 4. Compute comparative metrics
        raw_flat = raw_data.flatten().astype(np.float64)
        clean_flat = cleaned_data.flatten().astype(np.float64)

        duration = round(len(raw_flat) / sample_rate, 2)
        raw_rms = float(np.sqrt(np.mean(np.square(raw_flat))))
        clean_rms = float(np.sqrt(np.mean(np.square(clean_flat))))
        rms_reduction = round(((raw_rms - clean_rms) / max(raw_rms, 1e-5)) * 100, 1)

        raw_peak = int(np.max(np.abs(raw_flat)))
        clean_peak = int(np.max(np.abs(clean_flat)))

        raw_snr = self.estimate_snr(raw_data)
        clean_snr = self.estimate_snr(cleaned_data)
        snr_diff = round(clean_snr - raw_snr, 2)

        return NoiseReductionReport(
            raw_file_path=raw_path,
            cleaned_file_path=clean_path,
            sample_rate=sample_rate,
            duration_seconds=duration,
            raw_rms=round(raw_rms, 2),
            cleaned_rms=round(clean_rms, 2),
            rms_reduction_pct=rms_reduction,
            raw_peak=raw_peak,
            cleaned_peak=clean_peak,
            raw_snr_db=raw_snr,
            cleaned_snr_db=clean_snr,
            snr_improvement_db=snr_diff,
        )


# Convenience function for direct one-line usage
def reduce_noise_file(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    prop_decrease: float = 0.80,
) -> NoiseReductionReport:
    """Convenience helper to reduce noise from an audio file."""
    reducer = NoiseReducer(prop_decrease=prop_decrease)
    return reducer.clean_audio_file(input_file, output_file)
