"""Paper reproduction entrypoints (Table 1/2 + demos)."""

from __future__ import annotations

import json
from typing import Any


def print_console_json(prefix: str, title: str, payload: dict[str, Any]) -> None:
    print(f"[{prefix}] {title}:")
    print(json.dumps(payload, indent=2, sort_keys=True))

