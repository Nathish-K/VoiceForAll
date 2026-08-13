"""
Audio Analyzer Module (Level 2)
------------------------------
Responsible for loading WAV files, extracting metadata, computing amplitude
statistics (including RMS), generating time-domain waveforms, and computing
frequency-domain spectrograms.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless image saving
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal


@dataclass
class AudioAnalysisReport:
    """Structured report containing audio metrics and generated plot paths."""
    file_path: Path
    sample_rate: int
    num_samples: int
    duration_seconds: float
    num_channels: int
    data_type: str
    min_amplitude: int
    max_amplitude: int
    abs_max_amplitude: int
    mean_amplitude: float
    rms_amplitude: float
    waveform_path: Optional[Path] = None
    spectrogram_path: Optional[Path] = None


class AudioAnalyzer:
    """
    Audio analyzer for inspecting, calculating statistics, and visualizing
    digital audio signals.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            # Default to project root / audio_samples / analysis
            project_root = Path(__file__).resolve().parent.parent.parent
            self.output_dir = project_root / "audio_samples" / "analysis"

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_wav(file_path: Union[str, Path]) -> Tuple[int, np.ndarray]:
        """
        Loads a WAV file from disk and returns sample rate and raw samples.

        Args:
            file_path (str | Path): Path to the WAV file.

        Returns:
            Tuple[int, np.ndarray]: (sample_rate, audio_data)

        Raises:
            FileNotFoundError: If the WAV file does not exist.
            ValueError: If the file is not a valid WAV or is corrupt.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found at: {path}")

        try:
            sample_rate, audio_data = wavfile.read(str(path))
            return sample_rate, audio_data
        except Exception as exc:
            raise ValueError(f"Could not parse WAV file '{path}': {exc}") from exc

    @staticmethod
    def calculate_statistics(sample_rate: int, audio_data: np.ndarray) -> dict:
        """
        Calculates mathematical and amplitude statistics for audio samples.

        Args:
            sample_rate (int): Sampling rate in Hz.
            audio_data (np.ndarray): Audio samples.

        Returns:
            dict: Dictionary with computed audio statistics.
        """
        # Flatten to 1D if single-channel mono with shape (N, 1)
        flat_samples = audio_data.flatten().astype(np.float64)

        num_samples = len(flat_samples)
        duration_seconds = round(num_samples / sample_rate, 2)
        num_channels = audio_data.shape[1] if audio_data.ndim > 1 else 1

        min_amp = int(np.min(flat_samples))
        max_amp = int(np.max(flat_samples))
        abs_max_amp = int(np.max(np.abs(flat_samples)))
        mean_amp = float(np.mean(flat_samples))

        # RMS (Root Mean Square) Amplitude = sqrt(mean(samples^2))
        # Represents effective energy / perceived average loudness
        rms_amp = float(np.sqrt(np.mean(np.square(flat_samples))))

        return {
            "sample_rate": sample_rate,
            "num_samples": num_samples,
            "duration_seconds": duration_seconds,
            "num_channels": num_channels,
            "data_type": str(audio_data.dtype),
            "min_amplitude": min_amp,
            "max_amplitude": max_amp,
            "abs_max_amplitude": abs_max_amp,
            "mean_amplitude": mean_amp,
            "rms_amplitude": rms_amp,
        }

    def plot_waveform(
        self,
        sample_rate: int,
        audio_data: np.ndarray,
        filename: str = "waveform.png",
        title: str = "Audio Waveform (Time Domain)",
    ) -> Path:
        """
        Plots and saves the time-domain waveform of the audio.

        Args:
            sample_rate (int): Sampling rate in Hz.
            audio_data (np.ndarray): Audio samples.
            filename (str): Name of the output image file.
            title (str): Title for the plot.

        Returns:
            Path: Path to the saved waveform image.
        """
        flat_samples = audio_data.flatten()
        num_samples = len(flat_samples)

        # Create time axis in seconds: sample_index / sample_rate
        time_axis = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)

        # Calculate RMS for reference line
        rms_val = np.sqrt(np.mean(np.square(flat_samples.astype(np.float64))))

        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        ax.plot(time_axis, flat_samples, color="#2563eb", linewidth=0.8, label="Signal Amplitude")

        # Plot RMS threshold lines (+RMS and -RMS)
        ax.axhline(rms_val, color="#dc2626", linestyle="--", linewidth=1.2, label=f"+RMS ({rms_val:.1f})")
        ax.axhline(-rms_val, color="#dc2626", linestyle="--", linewidth=1.2, label=f"-RMS (-{rms_val:.1f})")
        ax.axhline(0, color="#6b7280", linestyle="-", linewidth=0.8, alpha=0.7)

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Amplitude (int16 PCM)", fontsize=10, fontweight="bold")
        ax.set_ylim(-32768, 32767)
        ax.set_xlim(0, time_axis[-1] if len(time_axis) > 0 else 1)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", framealpha=0.9)

        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path)
        plt.close(fig)

        return out_path

    def plot_spectrogram(
        self,
        sample_rate: int,
        audio_data: np.ndarray,
        filename: str = "spectrogram.png",
        title: str = "Audio Spectrogram (Frequency vs Time)",
    ) -> Path:
        """
        Computes STFT and plots the frequency-domain spectrogram.

        Args:
            sample_rate (int): Sampling rate in Hz.
            audio_data (np.ndarray): Audio samples.
            filename (str): Output filename.
            title (str): Plot title.

        Returns:
            Path: Path to saved spectrogram image.
        """
        flat_samples = audio_data.flatten().astype(np.float32)

        # Compute Short-Time Fourier Transform (STFT)
        # NPERSEG = 512 samples (~32 ms window at 16 kHz), NOVERLAP = 256 samples (50% overlap)
        nperseg = 512
        noverlap = 256
        frequencies, times, Sxx = signal.spectrogram(
            flat_samples,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="spectrum",
        )

        # Convert power to decibels (dB) for perceptual log-scale representation
        sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-10))

        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
        pcm = ax.pcolormesh(times, frequencies, sxx_db, shading="gouraud", cmap="magma")

        # Colorbar representing Energy/Intensity in dB
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Energy (dB)", fontsize=10, fontweight="bold")

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Frequency (Hz)", fontsize=10, fontweight="bold")
        ax.set_ylim(0, sample_rate // 2)  # 0 to 8000 Hz (Nyquist limit)

        # Highlight fundamental speech band (300 Hz - 3400 Hz)
        ax.axhspan(300, 3400, color="cyan", alpha=0.1, label="Core Speech Band (300-3400 Hz)")
        ax.legend(loc="upper right", framealpha=0.9)

        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path)
        plt.close(fig)

        return out_path

    def analyze(self, file_path: Union[str, Path]) -> AudioAnalysisReport:
        """
        Full analysis pipeline: loads WAV, computes statistics, generates waveform
        and spectrogram plots.

        Args:
            file_path (str | Path): Path to target WAV file.

        Returns:
            AudioAnalysisReport: Comprehensive analysis report object.
        """
        path = Path(file_path).resolve()
        sample_rate, audio_data = self.load_wav(path)
        stats = self.calculate_statistics(sample_rate, audio_data)

        # Create unique or standard plot names
        base_name = path.stem
        waveform_path = self.plot_waveform(
            sample_rate,
            audio_data,
            filename=f"{base_name}_waveform.png",
            title=f"Waveform: {path.name}",
        )
        spectrogram_path = self.plot_spectrogram(
            sample_rate,
            audio_data,
            filename=f"{base_name}_spectrogram.png",
            title=f"Spectrogram: {path.name}",
        )

        return AudioAnalysisReport(
            file_path=path,
            sample_rate=stats["sample_rate"],
            num_samples=stats["num_samples"],
            duration_seconds=stats["duration_seconds"],
            num_channels=stats["num_channels"],
            data_type=stats["data_type"],
            min_amplitude=stats["min_amplitude"],
            max_amplitude=stats["max_amplitude"],
            abs_max_amplitude=stats["abs_max_amplitude"],
            mean_amplitude=stats["mean_amplitude"],
            rms_amplitude=stats["rms_amplitude"],
            waveform_path=waveform_path,
            spectrogram_path=spectrogram_path,
        )

    def plot_comparison(
        self,
        raw_file_path: Union[str, Path],
        cleaned_file_path: Union[str, Path],
        filename: str = "before_after_comparison.png",
    ) -> Path:
        """
        Generates a 2x2 side-by-side comparative dashboard comparing raw noisy audio
        vs cleaned audio in both time domain (waveform) and frequency domain (spectrogram).

        Args:
            raw_file_path (str | Path): Raw WAV path.
            cleaned_file_path (str | Path): Cleaned WAV path.
            filename (str): Output filename.

        Returns:
            Path: Path to the generated comparison dashboard image.
        """
        raw_sr, raw_data = self.load_wav(raw_file_path)
        clean_sr, clean_data = self.load_wav(cleaned_file_path)

        raw_flat = raw_data.flatten()
        clean_flat = clean_data.flatten()

        raw_t = np.linspace(0, len(raw_flat) / raw_sr, len(raw_flat), endpoint=False)
        clean_t = np.linspace(0, len(clean_flat) / clean_sr, len(clean_flat), endpoint=False)

        fig, axs = plt.subplots(2, 2, figsize=(14, 8), dpi=120)

        # 1. Raw Waveform (Top Left)
        axs[0, 0].plot(raw_t, raw_flat, color="#dc2626", linewidth=0.7)
        axs[0, 0].set_title(f"BEFORE: Noisy Waveform ({Path(raw_file_path).name})", fontsize=11, fontweight="bold")
        axs[0, 0].set_ylabel("Amplitude (int16)", fontsize=9)
        axs[0, 0].set_ylim(-32768, 32767)
        axs[0, 0].grid(True, linestyle=":", alpha=0.6)

        # 2. Cleaned Waveform (Top Right)
        axs[0, 1].plot(clean_t, clean_flat, color="#16a34a", linewidth=0.7)
        axs[0, 1].set_title(f"AFTER: Cleaned Waveform ({Path(cleaned_file_path).name})", fontsize=11, fontweight="bold")
        axs[0, 1].set_ylabel("Amplitude (int16)", fontsize=9)
        axs[0, 1].set_ylim(-32768, 32767)
        axs[0, 1].grid(True, linestyle=":", alpha=0.6)

        # 3. Raw Spectrogram (Bottom Left)
        f_raw, t_raw, sxx_raw = signal.spectrogram(raw_flat.astype(np.float32), fs=raw_sr, nperseg=512, noverlap=256)
        sxx_raw_db = 10 * np.log10(np.maximum(sxx_raw, 1e-10))
        pcm1 = axs[1, 0].pcolormesh(t_raw, f_raw, sxx_raw_db, shading="gouraud", cmap="magma")
        axs[1, 0].set_title("BEFORE: Noisy Spectrogram", fontsize=11, fontweight="bold")
        axs[1, 0].set_xlabel("Time (seconds)", fontsize=9)
        axs[1, 0].set_ylabel("Frequency (Hz)", fontsize=9)
        axs[1, 0].set_ylim(0, raw_sr // 2)
        fig.colorbar(pcm1, ax=axs[1, 0], label="dB")

        # 4. Cleaned Spectrogram (Bottom Right)
        f_cln, t_cln, sxx_cln = signal.spectrogram(clean_flat.astype(np.float32), fs=clean_sr, nperseg=512, noverlap=256)
        sxx_cln_db = 10 * np.log10(np.maximum(sxx_cln, 1e-10))
        pcm2 = axs[1, 1].pcolormesh(t_cln, f_cln, sxx_cln_db, shading="gouraud", cmap="magma")
        axs[1, 1].set_title("AFTER: Cleaned Spectrogram", fontsize=11, fontweight="bold")
        axs[1, 1].set_xlabel("Time (seconds)", fontsize=9)
        axs[1, 1].set_ylabel("Frequency (Hz)", fontsize=9)
        axs[1, 1].set_ylim(0, clean_sr // 2)
        fig.colorbar(pcm2, ax=axs[1, 1], label="dB")

        plt.suptitle("LEVEL 3: AUDIO NOISE REDUCTION BEFORE & AFTER COMPARISON", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()

        comp_dir = self.output_dir / "comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)
        out_path = comp_dir / filename
        plt.savefig(out_path)
        plt.close(fig)

        return out_path


def analyze_audio_file(file_path: Union[str, Path]) -> AudioAnalysisReport:
    """Convenience helper function to analyze a WAV file in one call."""
    analyzer = AudioAnalyzer()
    return analyzer.analyze(file_path)
