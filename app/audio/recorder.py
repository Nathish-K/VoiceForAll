"""
Audio Recorder Module (Level 1)
------------------------------
Responsible for capturing raw microphone audio input using `sounddevice`,
storing it as a NumPy array, and saving it to disk as a standard WAV file
using `scipy.io.wavfile`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import math
from typing import Optional, Union
import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
import sounddevice as sd


@dataclass
class AudioMetadata:
    """Metadata describing an audio recording."""
    sample_rate: int
    num_samples: int
    duration_seconds: float
    num_channels: int
    file_path: Optional[Path] = None
    data_type: str = "int16"


class AudioRecorder:
    """
    Modular Audio Recorder for capturing microphone input cleanly.

    Attributes:
        sample_rate (int): Sampling rate in Hertz (samples/second). Default is 16000 (16 kHz).
        channels (int): Target number of audio channels (1 for Mono, 2 for Stereo). Default is 1.
        dtype (str): NumPy data type for audio samples. Default is 'int16'.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "int16",
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be a positive integer, got {sample_rate}")
        if channels not in (1, 2):
            raise ValueError(f"Channels must be 1 (mono) or 2 (stereo), got {channels}")
        if dtype not in ("int16", "float32"):
            raise ValueError(f"Unsupported dtype '{dtype}'. Supported: 'int16', 'float32'")

        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype

    def check_input_device(self) -> dict:
        """
        Validates that a working microphone/input device is available on the system.

        Returns:
            dict: Information about the default input device.

        Raises:
            RuntimeError: If no audio input device is detected or accessible.
        """
        try:
            device_info = sd.query_devices(kind="input")
            if not device_info:
                raise RuntimeError("No input audio device found on the system.")
            return device_info
        except sd.PortAudioError as exc:
            raise RuntimeError(
                f"Failed to access audio hardware: {exc}\n"
                "Please check microphone connection and operating system permissions."
            ) from exc

    @staticmethod
    def normalize_audio(audio_data: np.ndarray, target_peak_ratio: float = 0.90) -> np.ndarray:
        """
        Applies peak normalization so recorded speech audio has optimal volume.

        Args:
            audio_data (np.ndarray): Audio samples (int16 or float32).
            target_peak_ratio (float): Target peak fraction relative to max range (default: 0.90).

        Returns:
            np.ndarray: Peak-normalized audio data matching input dtype.
        """
        if audio_data is None or audio_data.size == 0:
            return audio_data

        original_dtype = audio_data.dtype
        float_data = audio_data.astype(np.float64)

        current_peak = np.max(np.abs(float_data))
        if current_peak < 1e-5:
            return audio_data  # Silence, avoid division by zero

        if original_dtype == np.int16:
            max_val = 32767.0
            target_peak = max_val * target_peak_ratio
            if current_peak < target_peak:
                gain = target_peak / current_peak
                # Limit maximum gain boost to avoid amplifying pure noise background (> 20x gain cap)
                gain = min(gain, 20.0)
                scaled = float_data * gain
                clipped = np.clip(scaled, -32768.0, 32767.0)
                return clipped.astype(np.int16)
        else:
            target_peak = target_peak_ratio
            if current_peak < target_peak:
                gain = target_peak / current_peak
                gain = min(gain, 20.0)
                scaled = float_data * gain
                clipped = np.clip(scaled, -1.0, 1.0)
                return clipped.astype(np.float32)

        return audio_data

    def record(self, duration: float, auto_normalize: bool = True) -> np.ndarray:
        """
        Records audio from default microphone at native hardware sample rate, then
        resamples to target sample rate (16 kHz mono) and applies gain normalization.

        Args:
            duration (float): Recording length in seconds (> 0).
            auto_normalize (bool): If True, applies peak gain normalization.

        Returns:
            np.ndarray: Recorded audio samples matching self.sample_rate and self.channels.
        """
        if duration <= 0:
            raise ValueError(f"Recording duration must be greater than 0 seconds, got {duration}")

        device_info = self.check_input_device()
        hw_sr = int(device_info.get("default_samplerate", self.sample_rate))
        hw_channels = int(device_info.get("max_input_channels", self.channels))
        hw_channels = max(1, min(hw_channels, 2))  # Use mono or stereo hardware capture

        hw_frames = int(duration * hw_sr)

        try:
            # Capture at hardware native rate & channel count to prevent driver distortion
            raw_buffer = sd.rec(
                frames=hw_frames,
                samplerate=hw_sr,
                channels=hw_channels,
                dtype=self.dtype,
            )
            sd.wait()

            # 1. Convert multi-channel to target mono/stereo
            float_buf = raw_buffer.astype(np.float32)
            if hw_channels > 1 and self.channels == 1:
                float_buf = np.mean(float_buf, axis=1, keepdims=True)
            elif hw_channels == 1 and self.channels == 2:
                float_buf = np.tile(float_buf, (1, 2))

            # 2. Resample to target sample_rate if hardware rate differs
            if hw_sr != self.sample_rate:
                target_num_samples = int(round(duration * self.sample_rate))
                # Use scipy.signal.resample_poly or resample
                resampled_channels = []
                for ch in range(float_buf.shape[1]):
                    ch_resampled = signal.resample(float_buf[:, ch], target_num_samples)
                    resampled_channels.append(ch_resampled)
                float_buf = np.column_stack(resampled_channels)

            # Convert back to requested dtype
            if self.dtype == "int16":
                clipped = np.clip(float_buf, -32768.0, 32767.0)
                audio_data = clipped.astype(np.int16)
            else:
                audio_data = float_buf.astype(np.float32)

            # 3. Peak gain normalization
            if auto_normalize:
                audio_data = self.normalize_audio(audio_data, target_peak_ratio=0.90)

            return audio_data

        except sd.PortAudioError as exc:
            raise RuntimeError(
                f"Audio recording failed: {exc}\n"
                "Ensure another application is not locking the microphone."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Unexpected error during audio recording: {exc}") from exc

    def save_wav(self, file_path: Union[str, Path], audio_data: np.ndarray) -> Path:
        """
        Saves a NumPy audio array to a standard WAV audio file.

        Args:
            file_path (str | Path): Destination file path (e.g. 'audio_samples/raw/rec.wav').
            audio_data (np.ndarray): NumPy array containing audio samples.

        Returns:
            Path: Resolved absolute Path of the saved file.
        """
        if audio_data is None or audio_data.size == 0:
            raise ValueError("Cannot save empty audio data.")

        target_path = Path(file_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            wavfile.write(
                filename=str(target_path),
                rate=self.sample_rate,
                data=audio_data,
            )
            return target_path
        except Exception as exc:
            raise IOError(f"Failed to write WAV file to '{target_path}': {exc}") from exc

    def get_metadata(
        self, audio_data: np.ndarray, file_path: Optional[Union[str, Path]] = None
    ) -> AudioMetadata:
        """Computes structured metadata from recorded audio data."""
        num_samples = audio_data.shape[0]
        duration_seconds = round(num_samples / self.sample_rate, 2)
        num_channels = audio_data.shape[1] if audio_data.ndim > 1 else 1
        path_obj = Path(file_path).resolve() if file_path else None

        return AudioMetadata(
            sample_rate=self.sample_rate,
            num_samples=num_samples,
            duration_seconds=duration_seconds,
            num_channels=num_channels,
            file_path=path_obj,
            data_type=str(audio_data.dtype),
        )


# Functional convenience wrappers
def record_audio(
    duration: float,
    sample_rate: int = 16000,
    channels: int = 1,
    dtype: str = "int16",
    auto_normalize: bool = True,
) -> np.ndarray:
    """Convenience function to record audio in a single call."""
    recorder = AudioRecorder(sample_rate=sample_rate, channels=channels, dtype=dtype)
    return recorder.record(duration, auto_normalize=auto_normalize)


def save_audio_to_wav(
    file_path: Union[str, Path],
    audio_data: np.ndarray,
    sample_rate: int = 16000,
) -> Path:
    """Convenience function to save audio data to a WAV file in a single call."""
    recorder = AudioRecorder(sample_rate=sample_rate)
    return recorder.save_wav(file_path=file_path, audio_data=audio_data)

