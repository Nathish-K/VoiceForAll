"""
Unit tests for AudioRecorder (Level 1)
--------------------------------------
Validates metadata extraction, parameter validation, and WAV writing using synthetic signals.
"""

import tempfile
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
import pytest

from app.audio.recorder import AudioRecorder, AudioMetadata


def test_recorder_initialization():
    """Verify default and custom initialization parameters."""
    rec = AudioRecorder(sample_rate=16000, channels=1, dtype="int16")
    assert rec.sample_rate == 16000
    assert rec.channels == 1
    assert rec.dtype == "int16"


def test_invalid_parameters():
    """Verify error handling on invalid initialization parameters."""
    with pytest.raises(ValueError):
        AudioRecorder(sample_rate=-100)
    with pytest.raises(ValueError):
        AudioRecorder(channels=5)
    with pytest.raises(ValueError):
        AudioRecorder(dtype="complex128")


def test_save_and_read_wav():
    """Verify synthetic 16 kHz tone generation, saving, and reading."""
    sample_rate = 16000
    duration = 1.0  # 1 second
    freq = 440.0  # 440 Hz standard A tone

    # Generate synthetic 16-bit PCM sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    synthetic_signal = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    synthetic_signal = synthetic_signal.reshape(-1, 1)

    rec = AudioRecorder(sample_rate=sample_rate, channels=1, dtype="int16")

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "test_tone.wav"
        saved_path = rec.save_wav(file_path=test_file, audio_data=synthetic_signal)

        assert saved_path.exists()

        # Read back with scipy to verify file integrity
        read_rate, read_data = wavfile.read(str(saved_path))
        assert read_rate == 16000
        assert read_data.shape[0] == 16000
        assert read_data.dtype == np.int16
        np.testing.assert_array_equal(synthetic_signal.flatten(), read_data.flatten())


def test_metadata_computation():
    """Verify metadata calculations (duration, samples, channels)."""
    sample_rate = 16000
    num_samples = 32000  # Exactly 2 seconds
    dummy_audio = np.zeros((num_samples, 1), dtype=np.int16)

    rec = AudioRecorder(sample_rate=sample_rate, channels=1)
    meta = rec.get_metadata(dummy_audio, file_path="dummy.wav")

    assert isinstance(meta, AudioMetadata)
    assert meta.sample_rate == 16000
    assert meta.num_samples == 32000
    assert meta.duration_seconds == 2.0
    assert meta.num_channels == 1
    assert meta.data_type == "int16"
