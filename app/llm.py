import asyncio
import os
import json
import logging
from typing import List

import groq as groq_lib
from groq import Groq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from app.schemas import KnowledgeUnit

logger = logging.getLogger(__name__)

# Groq model — fast & large context, great for structured extraction
MODEL_NAME = "qwen/qwen3.8-27b"


def get_client() -> Groq:
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Global state to track API limits across requests
API_STATUS = {
    "requests_remaining": "Unknown",
    "tokens_remaining": "Unknown",
}


# ---------------------------------------------------------------------------
# Retry-wrapped synchronous Groq call
# Retries on rate-limit (429), connection errors, and 5xx server errors.
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type(
        (
            groq_lib.RateLimitError,
            groq_lib.APIConnectionError,
            groq_lib.InternalServerError,
        )
    ),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_groq_sync(prompt: str) -> str:
    """Synchronous Groq chat completion with tenacity retry logic."""
    client = get_client()
    raw_response = client.chat.completions.with_raw_response.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert AI that extracts structured knowledge units from documents. "
                    "Always respond with valid JSON only, wrapped in a top-level object with a "
                    "\"units\" key containing the array."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        # json_object mode requires a JSON *object* (not array) at the top level
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=900,
    )
    
    # Extract rate limit headers from the raw HTTP response
    headers = raw_response.headers
    API_STATUS["requests_remaining"] = headers.get("x-ratelimit-remaining-requests", "Unknown")
    API_STATUS["tokens_remaining"] = headers.get("x-ratelimit-remaining-tokens", "Unknown")
    
    # Parse the actual response payload
    response = raw_response.parse()
    return response.choices[0].message.content


async def extract_knowledge_units(
    text: str, document_name: str, page_start: int, page_end: int
) -> List[KnowledgeUnit]:
    """
    Async entry point: dispatches the blocking Groq call to a thread pool via
    asyncio.to_thread() so FastAPI's event loop is never blocked.
    """
    prompt = f"""
    You are an expert AI tasked with extracting semantic knowledge units from the following text extracted from a document named "{document_name}" (pages {page_start} to {page_end}).

    A knowledge unit should represent a semantically meaningful piece of knowledge rather than an arbitrary token chunk.
    It should be comprehensive and self-contained.

    Types of knowledge units: concept, definition, algorithm, procedure, example, comparison, fact, reference.
    Relationship types: part_of, related_to, depends_on, contrasts_with.

    Text:
    '''
    {text}
    '''

    Extract the knowledge units and return a JSON object with a single key "units" whose value is an array of objects matching this schema:
    {{
      "units": [
        {{
          "id": "unique-slug-style-id",
          "title": "Title of the unit",
          "type": "one of the types above",
          "document": "{document_name}",
          "section": "Section name if identifiable from context, else null",
          "page_start": {page_start},
          "page_end": {page_end},
          "keywords": ["keyword1", "keyword2"],
          "content": "The actual detailed content formatted in Markdown",
          "relationships": [
            {{
              "type": "one of the relationship types above",
              "target_id": "target-unit-id"
            }}
          ]
        }}
      ]
    }}

    IMPORTANT: Return ONLY the JSON object described above. No markdown fences, no extra text.
    """

    # Offload the blocking HTTP call to a thread — keeps the event loop free
    raw_text = await asyncio.to_thread(_call_groq_sync, prompt)

    try:
        data = json.loads(raw_text)

        # Groq json_object mode returns a dict — extract the "units" array
        if isinstance(data, dict):
            items = data.get("units", [])
            # fallback: grab first list value if key is different
            if not items:
                for v in data.values():
                    if isinstance(v, list):
                        items = v
                        break
        else:
            items = data  # shouldn't happen with json_object mode, but guard anyway

        units = [KnowledgeUnit(**item) for item in items]
        return units
    except Exception as e:
        logger.error("Error parsing Groq output: %s", e)
        logger.debug("Raw output: %s", raw_text)
        return []
