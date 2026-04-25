"""
UMBRA Sentrix PII Guard — local confidential data detection and alert layer.
Intercepts all input before environment sees it. Zero network calls.
Returns structured result with severity, types found, and redacted text.
"""

import re
from dataclasses import dataclass, field
from typing import Any


class SentrixBlockException(Exception):
    def __init__(self, redacted_text: str, options: list[str], locations: list[dict]):
        self.redacted_text = redacted_text
        self.options = options
        self.locations = locations
        super().__init__(f"SENTRIX BLOCK: confidential data detected. Options: {options}")


PATTERNS: dict[str, tuple[str, str]] = {
    "AADHAR":       (r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b", "BLOCK"),
    "PAN":          (r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "BLOCK"),
    "VOTER_ID":     (r"\b[A-Z]{2,3}[0-9]{7}\b", "WARN"),
    "PASSPORT":     (r"\b[A-Z][0-9]{7}\b", "WARN"),
    "DRIVING_LIC":  (r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4,8}\b", "WARN"),
    "GSTIN":        (r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b", "BLOCK"),
    "IFSC":         (r"\b[A-Z]{4}0[A-Z0-9]{6}\b", "BLOCK"),
    "UPI":          (r"\b[\w.\-]+@[a-zA-Z]+\b", "BLOCK"),
    "CREDIT_CARD":  (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b", "BLOCK"),
    "BANK_ACCOUNT": (r"\b[0-9]{9,18}\b", "WARN"),
    "MOBILE_IN":    (r"\b(?:\+91[\s-]?)?[6-9][0-9]{9}\b", "BLOCK"),
    "EMAIL":        (r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b", "WARN"),
    "PINCODE":      (r"\b[1-9][0-9]{5}\b", "WARN"),
    "GPS":          (r"\b[-+]?(?:[1-8]?\d(?:\.\d+)?|90(?:\.0+)?),\s*[-+]?(?:1[0-7]\d(?:\.\d+)?|[1-9]?\d(?:\.\d+)?|180(?:\.0+)?)\b", "WARN"),
    "PASSWORD":     (r"(?:password|pwd|pass)\s*[:=]\s*\S+", "BLOCK"),
    "API_KEY":      (r"\b(?:sk-|pk-|AKIA)[A-Za-z0-9_\-]{16,}\b", "BLOCK"),
    "JWT":          (r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b", "BLOCK"),
    "SSH_KEY":      (r"-----BEGIN [A-Z ]+PRIVATE KEY-----", "BLOCK"),
    "BEARER":       (r"Bearer\s+[A-Za-z0-9_\-\.]{20,}", "BLOCK"),
    "DB_CONN":      (r"(?:mysql|postgresql|mongodb|redis):\/\/[^\s]+", "BLOCK"),
    "AWS_CRED":     (r"(?:aws_access_key|aws_secret)[^\n]*", "BLOCK"),
}

COMBO_BLOCK_THRESHOLD = 2  # any single PII + 2 or more others


def _luhn_check(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    odd = digits[-1::-2]
    even = [sum(divmod(d * 2, 10)) for d in digits[-2::-2]]
    return (sum(odd) + sum(even)) % 10 == 0


def run(text: str) -> dict[str, Any]:
    locations: list[dict] = []
    types_found: set[str] = set()

    for pii_type, (pattern, sev) in PATTERNS.items():
        for m in re.finditer(pattern, text, re.IGNORECASE):
            val = m.group()
            if pii_type == "CREDIT_CARD" and not _luhn_check(val.replace(" ", "")):
                continue
            locations.append({
                "type": pii_type,
                "start": m.start(),
                "end": m.end(),
                "value_preview": val[:4] + "****",
                "severity": sev,
            })
            types_found.add(pii_type)

    # Combination risk escalation
    combo_block = False
    t = list(types_found)
    if (("AADHAR" in t or "PAN" in t or "VOTER_ID" in t) and "EMAIL" in t) or \
       ("MOBILE_IN" in t and "EMAIL" in t and "PASSWORD" in t) or \
       ("BANK_ACCOUNT" in t and "IFSC" in t) or \
       (len(t) >= COMBO_BLOCK_THRESHOLD + 1):
        combo_block = True

    has_block = combo_block or any(
        PATTERNS[lt][1] == "BLOCK" for lt in types_found if lt in PATTERNS
    )
    has_warn = not has_block and bool(types_found)

    severity = "block" if has_block else ("warn" if has_warn else "pass")

    # Build redacted text
    redacted = text
    for loc in sorted(locations, key=lambda x: x["start"], reverse=True):
        redacted = redacted[:loc["start"]] + f"[REDACTED-{loc['type']}]" + redacted[loc["end"]:]

    options = [
        "[1] Edit your prompt manually",
        "[2] Auto-redact (replace with [REDACTED-TYPE])",
        "[3] Cancel send completely",
    ] if severity == "block" else []

    result = {
        "pii_found": bool(types_found),
        "types": list(types_found),
        "severity": severity,
        "redacted_text": redacted,
        "locations": locations,
        "user_options": options,
    }

    if severity == "block":
        raise SentrixBlockException(redacted, options, locations)

    return result
