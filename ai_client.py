"""Shared Gemini client and helpers for EduGenie."""

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load variables from .env
load_dotenv()


# Use the model from .env.
# If GEMINI_MODEL is missing, use the lightweight model.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


def _get_api_key() -> str:
    """Get the Gemini API key from environment variables."""

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not key:
        raise RuntimeError(
            "Gemini API key is not configured. "
            "Set GEMINI_API_KEY in your environment or .env file."
        )

    return key


def get_client() -> genai.Client:
    """Create and return a Gemini client."""

    return genai.Client(api_key=_get_api_key())


def _generate_with_retry(
    prompt: str,
    *,
    system_instruction: str | None = None,
    temperature: float = 0.4,
    max_output_tokens: int = 2048,
    response_mime_type: str | None = None,
):
    """Generate Gemini content with retries for temporary errors."""

    client = get_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        response_mime_type=response_mime_type,
    )

    # Try the configured model first, then a lightweight fallback.
    models_to_try = [
        MODEL_NAME,
        "gemini-3.5-flash-lite",
    ]

    # Remove duplicates while keeping order.
    models_to_try = list(dict.fromkeys(models_to_try))

    last_error = None

    for model in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )

                text = getattr(response, "text", None)

                if not text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return response

            except Exception as exc:
                last_error = exc

                error_text = str(exc).lower()

                temporary_error = any(
                    code in error_text
                    for code in [
                        "429",
                        "500",
                        "503",
                        "504",
                        "unavailable",
                        "overloaded",
                        "high demand",
                        "temporarily",
                    ]
                )

                if not temporary_error:
                    raise

                # Exponential backoff: 2s, 4s, 8s
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))

        # Try the fallback model after retries.

    raise RuntimeError(
        "Gemini is temporarily unavailable. "
        "Please try again in a moment. "
        f"Last error: {last_error}"
    )


def generate_text(
    prompt: str,
    *,
    system_instruction: str | None = None,
    temperature: float = 0.4,
    max_output_tokens: int = 2048,
) -> str:
    """Generate normal text using Gemini."""

    response = _generate_with_retry(
        prompt,
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    text = getattr(response, "text", None)

    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    return text.strip()


def generate_json(
    prompt: str,
    *,
    system_instruction: str | None = None,
    max_output_tokens: int = 4096,
) -> Any:
    """Generate and parse JSON using Gemini."""

    response = _generate_with_retry(
        prompt,
        system_instruction=system_instruction,
        temperature=0.2,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
    )

    text = getattr(response, "text", None)

    if not text:
        raise RuntimeError(
            "Gemini returned an empty JSON response."
        )

    cleaned = text.strip()

    # Remove Markdown code fences if Gemini adds them.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Gemini returned invalid JSON: "
            f"{exc}. Raw response: {text[:1000]}"
        ) from exc