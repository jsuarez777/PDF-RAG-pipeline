#!/usr/bin/env python3
import os
import shlex
import sys
from pathlib import Path
from typing import Optional

from openai import OpenAI

from openai_client.pricing import PRICES


def load_env_var_from_profile(var_name: str, profile_path: Optional[Path] = None) -> Optional[str]:
    """Parse a shell profile for an `export VAR=value` (or `VAR=value`) line.

    Safely reads the file as text without executing it. Returns the value of
    the last matching assignment, with surrounding quotes stripped, or None.
    """
    if profile_path is None:
        profile_path = Path.home() / ".profile"
    if not profile_path.is_file():
        return None

    value: Optional[str] = None
    for raw in profile_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        name, sep, rest = line.partition("=")
        if not sep or name.strip() != var_name:
            continue
        rest = rest.strip()
        try:
            # shlex handles quotes and strips inline comments the way a shell would.
            parts = shlex.split(rest, comments=True)
            value = parts[0] if parts else ""
        except ValueError:
            value = rest.strip("'\"")
    return value


class MyOpenAIClient:
    """
    Factory to produce a configured OpenAI client.
    Reads OPENAI_API_KEY and OPENAI_ORG from the environment if not provided.
    """

    def __init__(
        self, model: str, api_key: Optional[str] = None, temperature: Optional[float] = None
    ):
        if not model:
            print("Error: 'model' is required to initialize MyOpenAIClient.", file=sys.stderr)
            raise ValueError("'model' parameter is required.")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # Fall back to ~/.profile if no key is defined in the environment.
            self.api_key = load_env_var_from_profile("OPENAI_API_KEY")
            if self.api_key:
                # Expose it so the OpenAI SDK and any child code can see it too.
                os.environ.setdefault("OPENAI_API_KEY", self.api_key)
        self._client: Optional[OpenAI] = None
        self.model: str = model
        self._temperature: Optional[float] = temperature
        self.pricing: Optional[dict] = PRICES.get(model)

    @property
    def temperature(self) -> Optional[float]:
        return self._temperature

    @temperature.setter
    def temperature(self, value: Optional[float]) -> None:
        self._temperature = value

    def validate_api_key(self) -> None:
        """Raise an error if API key is missing."""
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it before running: export OPENAI_API_KEY='sk-...'"
            )

    def get_client(self) -> OpenAI:
        if self._client is None:
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            # Disable OpenAI SDK's built-in retries - we handle retries at application level
            kwargs["max_retries"] = 0
            self._client = OpenAI(**kwargs)
        return self._client

    @staticmethod
    def available_models() -> dict[str, dict]:
        """Return models and their pricing info from the pricing CSV."""
        return dict(PRICES)

    def query(self, *, input: str | list, model: Optional[str] = None, **kwargs):
        """Convenience wrapper that uses the configured default model unless overridden.

        Parameters:
            input: The prompt string or messages to send to the responses API.
            model: Optional model name to override the factory default.
            **kwargs: Passed through to `client.responses.create`.
        """
        client = self.get_client()
        model_to_use = model or self.model
        if not model_to_use:
            raise ValueError(
                "No model specified: pass `model=` or set `model` when constructing MyOpenAIClient."
            )
        if self._temperature is not None:
            kwargs.setdefault("temperature", self._temperature)
        return client.responses.create(model=model_to_use, input=input, **kwargs)
