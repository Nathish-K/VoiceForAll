"""
Unit tests for NoiseReducer (Level 3)
------------------------------------
Validates noise reduction on synthetic noisy speech signals, SNR estimation,
and before/after comparison dashboard generation.
"""

import tempfile
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
import pytest

from app.audio.noise_reducer import NoiseReducer, NoiseReductionReport
from app.audio.analyzer import AudioAnalyzer


def test_snr_estimation():
    """Verify SNR estimation on clean vs noisy synthetic signals."""
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Clean signal: Speech-like burst in middle 1 second
    clean_speech = np.zeros_like(t)
    speech_indices = (t >= 0.5) & (t <= 1.5)
    clean_speech[speech_indices] = np.sin(2 * np.pi * 500 * t[speech_indices]) * 15000

    # Add Gaussian noise
    np.random.seed(42)
    noise = np.random.normal(0, 1500, len(t))
    noisy_signal = (clean_speech + noise).astype(np.int16)

    reducer = NoiseReducer()
    initial_snr = reducer.estimate_snr(noisy_signal)
    assert initial_snr > 0  # Signal clearly has identifiable speech vs noise


def test_noise_reduction_pipeline():
    """Verify noise reduction decreases noise floor and improves SNR."""
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # First 0.5s is pure background noise, next 1.0s is speech + noise, last 0.5s is noise
    clean_speech = np.zeros_like(t)
    speech_mask = (t >= 0.5) & (t <= 1.5)
    clean_speech[speech_mask] = np.sin(2 * np.pi * 600 * t[speech_mask]) * 18000

    np.random.seed(123)
    steady_fan_noise = (np.sin(2 * np.pi * 120 * t) + np.random.normal(0, 0.4, len(t))) * 2500
    noisy_signal = (clean_speech + steady_fan_noise).astype(np.int16)

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_path = Path(tmp_dir) / "test_fan_noisy.wav"
        clean_path = Path(tmp_dir) / "test_fan_cleaned.wav"

        wavfile.write(str(raw_path), sample_rate, noisy_signal)

        reducer = NoiseReducer(prop_decrease=0.80)
        report = reducer.clean_audio_file(raw_file_path=raw_path, output_file_path=clean_path)

        assert isinstance(report, NoiseReductionReport)
        assert clean_path.exists()

        # Cleaned audio should have lower noise floor RMS
        assert report.cleaned_rms < report.raw_rms
        assert report.rms_reduction_pct > 0
        assert report.snr_improvement_db >= 0  # SNR improved or preserved

        # Generate comparison plot
        analyzer = AudioAnalyzer(output_dir=Path(tmp_dir) / "analysis")
        comp_plot = analyzer.plot_comparison(raw_path, clean_path)
        assert comp_plot.exists()
