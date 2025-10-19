from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

from src.modules.helper.helper import Helper

load_dotenv(dotenv_path=os.path.join(Helper.ROOT, ".env"), override=False)


class SlackService:
    def __init__(self) -> None:
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self._webhook_url: str = webhook_url

    def send(self, text: str) -> bool:
        payload: dict[str, Any] = {"text": text}
        res = requests.post(self._webhook_url, json=payload, timeout=10)
        return res.ok
