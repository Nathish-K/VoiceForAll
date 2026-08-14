"""
Tests for Level 5 NLP Framework Component Structure
---------------------------------------------------
Verifies intent definitions, command parser serialization, classifier defaults,
and entity extractor interface contracts.
"""

import pytest
from app.nlp.intent_definitions import Intent, ALL_INTENTS, CANONICAL_INTENTS
from app.nlp.command_parser import StructuredCommand, CommandParser
from app.nlp.intent_classifier import IntentClassifier
from app.nlp.entity_extractor import EntityExtractor


def test_intent_definitions():
    """Verifies all 12 required intents are defined."""
    required_intents = [
        "OPEN_APP",
        "CLOSE_APP",
        "PLAY_MUSIC",
        "PAUSE_MUSIC",
        "RESUME_MUSIC",
        "CALL_CONTACT",
        "SEND_MESSAGE",
        "WEB_SEARCH",
        "GET_WEATHER",
        "TAKE_SCREENSHOT",
        "UNKNOWN",
        "MULTI_COMMAND",
    ]
    for intent_name in required_intents:
        assert hasattr(Intent, intent_name)
        assert Intent[intent_name].value == intent_name
        assert Intent.is_valid(intent_name)

    canonical = Intent.get_canonical_intents()
    assert Intent.UNKNOWN not in canonical
    assert Intent.MULTI_COMMAND not in canonical
    assert len(canonical) == 10


def test_structured_command_serialization():
    """Verifies StructuredCommand dictionary and JSON serialization."""
    cmd = StructuredCommand(
        raw_text="open Spotify",
        intent=Intent.OPEN_APP,
        entities={"app": "Spotify"},
        confidence=0.95,
    )
    cmd_dict = cmd.to_dict()
    assert cmd_dict["raw_text"] == "open Spotify"
    assert cmd_dict["intent"] == "OPEN_APP"
    assert cmd_dict["entities"] == {"app": "Spotify"}
    assert cmd_dict["confidence"] == 0.95

    json_str = cmd.to_json()
    assert '"intent": "OPEN_APP"' in json_str

    deserialized = StructuredCommand.from_dict(cmd_dict)
    assert deserialized.intent == Intent.OPEN_APP
    assert deserialized.raw_text == "open Spotify"
    assert deserialized.entities == {"app": "Spotify"}


def test_intent_classifier_defaults():
    """Verifies default behavior of untrained IntentClassifier."""
    classifier = IntentClassifier(confidence_threshold=0.60)
    assert not classifier.is_trained
    intent, conf = classifier.predict("open Spotify")
    assert intent == Intent.UNKNOWN
    assert conf == 0.0

    # Empty string handling
    intent_empty, conf_empty = classifier.predict("")
    assert intent_empty == Intent.UNKNOWN
    assert conf_empty == 0.0


def test_entity_extractor_defaults():
    """Verifies default behavior of untrained EntityExtractor."""
    extractor = EntityExtractor()
    assert not extractor.is_trained
    entities = extractor.extract("open Spotify", intent=Intent.OPEN_APP)
    assert entities == {}


def test_command_parser():
    """Verifies CommandParser parsing flow."""
    parser = CommandParser()
    parsed = parser.parse("open Spotify")
    assert isinstance(parsed, StructuredCommand)
    assert parsed.intent == Intent.UNKNOWN
    assert parsed.raw_text == "open Spotify"
