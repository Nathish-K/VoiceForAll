"""
Unit tests for AudioAnalyzer (Level 2)
--------------------------------------
Validates statistics computation (min, max, mean, RMS), waveform plotting,
and spectrogram generation using synthetic multi-frequency test signals.
"""

import tempfile
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
import pytest

from app.audio.analyzer import AudioAnalyzer, AudioAnalysisReport


def test_statistics_calculation():
    """Verify statistical and RMS amplitude calculations on a known signal."""
    sample_rate = 16000
    duration = 1.0
    # Generate a pure sine wave with amplitude 10,000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    amplitude = 10000
    signal_data = (np.sin(2 * np.pi * 440 * t) * amplitude).astype(np.int16)

    stats = AudioAnalyzer.calculate_statistics(sample_rate, signal_data)

    assert stats["sample_rate"] == 16000
    assert stats["num_samples"] == 16000
    assert stats["duration_seconds"] == 1.0
    assert stats["num_channels"] == 1
    assert stats["data_type"] == "int16"
    assert abs(stats["abs_max_amplitude"] - 10000) <= 2  # Close to peak
    assert abs(stats["mean_amplitude"]) < 5  # Centered around 0

    # Theoretical RMS of sine wave with peak A is A / sqrt(2) ≈ 0.7071 * A
    expected_rms = amplitude / np.sqrt(2)
    assert abs(stats["rms_amplitude"] - expected_rms) < 50


def test_analyzer_end_to_end():
    """Verify loading WAV, computing statistics, and generating PNG plots."""
    sample_rate = 16000
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    # Composite signal: 500 Hz (voice fundamental) + 2000 Hz (formant)
    composite = ((np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 2000 * t)) * 10000).astype(np.int16)

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "test_voice_synth.wav"
        wavfile.write(str(wav_path), sample_rate, composite)

        analyzer = AudioAnalyzer(output_dir=Path(tmp_dir) / "analysis")
        report = analyzer.analyze(wav_path)

        assert isinstance(report, AudioAnalysisReport)
        assert report.sample_rate == 16000
        assert report.num_samples == 16000
        assert report.duration_seconds == 1.0
        assert report.waveform_path is not None and report.waveform_path.exists()
        assert report.spectrogram_path is not None and report.spectrogram_path.exists()
        assert report.waveform_path.stat().st_size > 1000
        assert report.spectrogram_path.stat().st_size > 1000
