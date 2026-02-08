from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StateImage:
    mime_type: str
    data_base64: str
    data_url: str
    width: int
    height: int
