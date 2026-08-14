"""
Intent Classifier Interface (Level 5)
-------------------------------------
Defines the architectural contract for classifying user transcription text into
one of the canonical intents defined in `intent_definitions.py`.

Design Notes:
- Input: Raw text string (from Level 4 Whisper ASR output).
- Output: Tuple of (Intent, confidence_score).
- Low-confidence or out-of-domain predictions map to `Intent.UNKNOWN`.
- ML model training and dataset creation will occur in subsequent project stages.
"""

from typing import Optional, Tuple, Union
from app.nlp.intent_definitions import Intent, CANONICAL_INTENTS


class IntentClassifier:
    """
    Modular Intent Classifier interface for voice commands.

    Attributes:
        confidence_threshold (float): Minimum confidence required before assigning
            a canonical intent. Predictions below threshold return Intent.UNKNOWN.
        is_trained (bool): Status flag indicating whether model weights are loaded.
    """

    def __init__(self, confidence_threshold: float = 0.60) -> None:
        self.confidence_threshold = confidence_threshold
        self.is_trained = False

    def predict(self, text: str) -> Tuple[Intent, float]:
        """
        Classifies transcription text into an Intent and confidence score.

        Args:
            text (str): Input voice transcription text string.

        Returns:
            Tuple[Intent, float]: Recognized intent enum and confidence score (0.0 - 1.0).
        """
        if not text or not text.strip():
            return Intent.UNKNOWN, 0.0

        if not self.is_trained:
            # Untrained skeleton defaults to UNKNOWN with 0.0 confidence
            return Intent.UNKNOWN, 0.0

        # TODO: Implement model inference logic in future training step
        return Intent.UNKNOWN, 0.0

    def load_model(self, model_path: str) -> None:
        """
        Loads pre-trained classification model weights/artifacts from disk.

        Args:
            model_path (str): Path to saved model file/directory under `models/`.
        """
        # TODO: Implement model loader in future training step
        pass
