import json
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_pattern = r"\{.*?\}|\[.*?\]"
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                match = match.replace("'", '"')
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        logger.warning("No valid JSON found in the text.")
        return None


def parse_text_find_number(text) -> Optional[int]:
    item = parse_json(text)
    try:
        match = int(item["final_answer"])
        if match in range(-1, 5):
            return match
    except (ValueError, KeyError, TypeError):
        logger.error("Error parsing number from text.")
    return None


def parse_text_find_confidence(text):
    item = parse_json(text)
    try:
        return float(item["confidence"])
    except (ValueError, KeyError, TypeError):
        logger.error("Error parsing confidence from text.")
    return None
