"""
Entity Extractor Interface (Level 5)
------------------------------------
Defines the architectural contract for extracting slot entities (e.g. app name,
contact name, search query, location) from user transcription text.

Design Notes:
- Input: Raw text string and optional predicted Intent.
- Output: Dictionary mapping entity slot names to extracted values (e.g. {"app": "Spotify"}).
- Rule-based regex pattern matchers or slot-filling ML models will be integrated
  in subsequent project stages.
"""

from typing import Any, Dict, Optional
from app.nlp.intent_definitions import Intent


class EntityExtractor:
    """
    Modular Entity Extractor interface for extracting intent-specific parameters.

    Attributes:
        is_trained (bool): Status flag indicating whether slot extraction rules/models are ready.
    """

    def __init__(self) -> None:
        self.is_trained = False

    def extract(self, text: str, intent: Optional[Intent] = None) -> Dict[str, Any]:
        """
        Extracts slot entities from user transcription text.

        Args:
            text (str): Input transcription text string.
            intent (Optional[Intent]): Pre-classified intent context to guide slot extraction.

        Returns:
            Dict[str, Any]: Extracted entity slots key-value mapping.
        """
        if not text or not text.strip():
            return {}

        # TODO: Implement rule-based or model-based slot extraction in future steps
        return {}
