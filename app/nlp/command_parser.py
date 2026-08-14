"""
Structured Command Parser (Level 5)
-----------------------------------
Defines the final structured output contract of Level 5 NLP understanding.
Encapsulates intent, extracted entities, confidence score, and raw transcription text.

Note: This module is an UNDERSTANDING layer only. It does NOT execute commands
or interact with external systems/runtimes.
"""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, Optional, Union
from app.nlp.intent_definitions import Intent


@dataclass
class StructuredCommand:
    """
    Structured representation of an analyzed user voice command.

    Attributes:
        raw_text (str): Original transcribed text from Level 4 Whisper ASR.
        intent (Intent): Categorized user intent (e.g. Intent.OPEN_APP).
        entities (Dict[str, Any]): Map of extracted entity slots (e.g. {"app": "Spotify"}).
        confidence (float): Classification confidence score between 0.0 and 1.0.
        is_composite (bool): True if utterance contains multiple commands.
    """

    raw_text: str
    intent: Intent
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    is_composite: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the structured command into a standard Python dictionary."""
        return {
            "raw_text": self.raw_text,
            "intent": self.intent.value if isinstance(self.intent, Intent) else str(self.intent),
            "entities": self.entities,
            "confidence": round(self.confidence, 4),
            "is_composite": self.is_composite,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serializes the structured command into a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredCommand":
        """Instantiates a StructuredCommand from a dictionary payload."""
        intent_raw = data.get("intent", Intent.UNKNOWN.value)
        intent_enum = (
            Intent(intent_raw)
            if Intent.is_valid(intent_raw)
            else Intent.UNKNOWN
        )
        return cls(
            raw_text=data.get("raw_text", ""),
            intent=intent_enum,
            entities=data.get("entities", {}),
            confidence=float(data.get("confidence", 1.0)),
            is_composite=bool(data.get("is_composite", False)),
        )


class CommandParser:
    """
    Combines Intent Classifier and Entity Extractor predictions into a
    unified StructuredCommand object.
    """

    def __init__(self, classifier: Any = None, extractor: Any = None) -> None:
        self.classifier = classifier
        self.extractor = extractor

    def parse(self, text: str) -> StructuredCommand:
        """
        Parses raw transcription text into a StructuredCommand.

        Args:
            text (str): Transcribed speech string from Level 4 Whisper.

        Returns:
            StructuredCommand: Validated structured command payload.
        """
        clean_text = text.strip() if text else ""
        if not clean_text:
            return StructuredCommand(
                raw_text="",
                intent=Intent.UNKNOWN,
                entities={},
                confidence=0.0,
            )

        # High-level pipeline flow (To be wired with trained components in future steps)
        # 1. Classify intent
        intent = Intent.UNKNOWN
        confidence = 0.0
        if self.classifier and hasattr(self.classifier, "predict"):
            intent, confidence = self.classifier.predict(clean_text)

        # 2. Extract entities
        entities: Dict[str, Any] = {}
        if self.extractor and hasattr(self.extractor, "extract"):
            entities = self.extractor.extract(clean_text, intent=intent)

        return StructuredCommand(
            raw_text=clean_text,
            intent=intent,
            entities=entities,
            confidence=confidence,
        )
