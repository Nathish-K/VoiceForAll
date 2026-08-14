"""
Centralized Intent Definitions (Level 5)
----------------------------------------
Defines the standard set of canonical intents for the local voice assistant.
All modules in Level 5 must use these centralized definitions rather than raw strings.
"""

from enum import Enum, unique
from typing import List, Set


@unique
class Intent(str, Enum):
    """
    Canonical intent enumeration for voice assistant interactions.

    Intents:
        OPEN_APP: Launch a local or system application (e.g. "open Spotify").
        CLOSE_APP: Close/terminate an active application (e.g. "close Chrome").
        PLAY_MUSIC: Play a song, playlist, or artist (e.g. "play jazz music").
        PAUSE_MUSIC: Pause active audio/music playback (e.g. "pause music").
        RESUME_MUSIC: Resume paused audio/music playback (e.g. "resume playback").
        CALL_CONTACT: Initiate a voice call to a contact (e.g. "call Alice").
        SEND_MESSAGE: Send a message to a contact (e.g. "send message to Bob").
        WEB_SEARCH: Perform a web search query (e.g. "search for quantum computing").
        GET_WEATHER: Query weather information (e.g. "what's the weather in Tokyo").
        TAKE_SCREENSHOT: Capture a screenshot of the current screen.
        UNKNOWN: Inference fallback for unrecognized or low-confidence utterances.
        MULTI_COMMAND: Composite command containing multiple intents (e.g. "open Spotify and play jazz").
    """

    OPEN_APP = "OPEN_APP"
    CLOSE_APP = "CLOSE_APP"
    PLAY_MUSIC = "PLAY_MUSIC"
    PAUSE_MUSIC = "PAUSE_MUSIC"
    RESUME_MUSIC = "RESUME_MUSIC"
    CALL_CONTACT = "CALL_CONTACT"
    SEND_MESSAGE = "SEND_MESSAGE"
    WEB_SEARCH = "WEB_SEARCH"
    GET_WEATHER = "GET_WEATHER"
    TAKE_SCREENSHOT = "TAKE_SCREENSHOT"
    UNKNOWN = "UNKNOWN"
    MULTI_COMMAND = "MULTI_COMMAND"

    @classmethod
    def get_canonical_intents(cls) -> Set["Intent"]:
        """
        Returns the set of standard target intent categories for model classification.
        Excludes UNKNOWN and MULTI_COMMAND from standard classification targets.
        """
        return {
            intent
            for intent in cls
            if intent not in (cls.UNKNOWN, cls.MULTI_COMMAND)
        }

    @classmethod
    def is_valid(cls, intent_str: str) -> bool:
        """Validates whether a given string corresponds to a defined intent."""
        return intent_str in cls._value2member_map_


# Canonical list of string representations for external configuration or dataset validation
ALL_INTENTS: List[str] = [intent.value for intent in Intent]
CANONICAL_INTENTS: List[str] = [
    intent.value for intent in Intent.get_canonical_intents()
]
