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
import numpy as np
import scipy.io.wavfile as wavfile
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
    Modular Audio Recorder for capturing microphone input.

    Attributes:
        sample_rate (int): Sampling rate in Hertz (samples/second). Default is 16000 (16 kHz).
        channels (int): Number of audio channels (1 for Mono, 2 for Stereo). Default is 1 (Mono).
        dtype (str): NumPy data type for audio samples. Default is 'int16' (16-bit signed integer).
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
            # Query the default input device configured in the operating system
            device_info = sd.query_devices(kind="input")
            if not device_info:
                raise RuntimeError("No input audio device found on the system.")
            return device_info
        except sd.PortAudioError as exc:
            raise RuntimeError(
                f"Failed to access audio hardware: {exc}\n"
                "Please check microphone connection and operating system permissions."
            ) from exc

    def record(self, duration: float) -> np.ndarray:
        """
        Records audio from the default microphone for the given duration.

        Args:
            duration (float): Recording length in seconds (must be > 0).

        Returns:
            np.ndarray: Recorded audio samples as a NumPy array with shape (num_samples, channels).

        Raises:
            ValueError: If duration is not positive.
            RuntimeError: If microphone access fails or recording encounters an error.
        """
        if duration <= 0:
            raise ValueError(f"Recording duration must be greater than 0 seconds, got {duration}")

        # Ensure a microphone is available before starting
        self.check_input_device()

        # Calculate total number of discrete samples needed
        num_frames = int(duration * self.sample_rate)

        try:
            # Start asynchronous recording into an internal buffer
            audio_buffer = sd.rec(
                frames=num_frames,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
            )

            # Block until recording finishes
            sd.wait()

            return audio_buffer

        except sd.PortAudioError as exc:
            raise RuntimeError(
                f"Audio recording failed: {exc}\n"
                "Ensure another application is not exclusively locking the microphone."
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

        Raises:
            ValueError: If audio_data is empty or invalid.
            IOError: If saving the file to disk fails.
        """
        if audio_data is None or audio_data.size == 0:
            raise ValueError("Cannot save empty audio data.")

        target_path = Path(file_path).resolve()

        # Create parent directory if it does not already exist
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Write WAV file using scipy.io.wavfile
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
        """
        Computes structured metadata from recorded audio data.

        Args:
            audio_data (np.ndarray): Audio samples.
            file_path (str | Path, optional): Path where the audio was saved.

        Returns:
            AudioMetadata: Summary information about the recording.
        """
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


# Functional convenience wrappers for quick one-line usage
def record_audio(
    duration: float,
    sample_rate: int = 16000,
    channels: int = 1,
    dtype: str = "int16",
) -> np.ndarray:
    """Convenience function to record audio in a single call."""
    recorder = AudioRecorder(sample_rate=sample_rate, channels=channels, dtype=dtype)
    return recorder.record(duration)


def save_audio_to_wav(
    file_path: Union[str, Path],
    audio_data: np.ndarray,
    sample_rate: int = 16000,
) -> Path:
    """Convenience function to save audio data to a WAV file in a single call."""
    recorder = AudioRecorder(sample_rate=sample_rate)
    return recorder.save_wav(file_path=file_path, audio_data=audio_data)
