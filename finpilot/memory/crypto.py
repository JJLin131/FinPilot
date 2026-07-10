from __future__ import annotations

import base64
import hashlib
import json
import os
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ENCRYPTED_PREFIX = "enc:v1:"
ASSOCIATED_DATA = b"finpilot-memory-v1"
NONCE_BYTES = 12


class MemoryCipher:
    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("Memory encryption key must decode to exactly 32 bytes.")
        self._cipher = AESGCM(key)
        self.key_fingerprint = hashlib.sha256(key).hexdigest()

    @classmethod
    def from_base64_key(cls, encoded_key: str) -> "MemoryCipher":
        if not encoded_key.strip():
            raise ValueError("MEMORY_ENCRYPTION_KEY is required.")
        try:
            key = base64.urlsafe_b64decode(encoded_key.encode("ascii"))
        except (ValueError, UnicodeEncodeError) as exc:
            raise ValueError("MEMORY_ENCRYPTION_KEY must be URL-safe base64.") from exc
        return cls(key)

    def encrypt_text(self, value: str) -> str:
        nonce = os.urandom(NONCE_BYTES)
        ciphertext = self._cipher.encrypt(nonce, value.encode("utf-8"), ASSOCIATED_DATA)
        payload = base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")
        return ENCRYPTED_PREFIX + payload

    def decrypt_text(self, value: str) -> str:
        if not value.startswith(ENCRYPTED_PREFIX):
            raise ValueError("Memory value is not encrypted.")
        encoded = value.removeprefix(ENCRYPTED_PREFIX)
        try:
            payload = base64.urlsafe_b64decode(encoded.encode("ascii"))
            nonce, ciphertext = payload[:NONCE_BYTES], payload[NONCE_BYTES:]
            if len(nonce) != NONCE_BYTES or not ciphertext:
                raise ValueError("Encrypted memory payload is incomplete.")
            return self._cipher.decrypt(nonce, ciphertext, ASSOCIATED_DATA).decode("utf-8")
        except Exception as exc:
            raise ValueError("Encrypted memory payload could not be decrypted.") from exc

    def decrypt_legacy_text(self, value: str) -> str:
        return self.decrypt_text(value) if value.startswith(ENCRYPTED_PREFIX) else value

    def encrypt_json(self, value: dict[str, Any]) -> str:
        return self.encrypt_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))

    def decrypt_json(self, value: str) -> dict[str, Any]:
        payload = json.loads(self.decrypt_text(value))
        if not isinstance(payload, dict):
            raise ValueError("Encrypted memory JSON must contain an object.")
        return payload
