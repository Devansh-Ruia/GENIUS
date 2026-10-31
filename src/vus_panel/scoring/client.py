"""CPU-only synchronous client. No evaluation, service, or GPU imports."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from pydantic import ValidationError

from vus_panel.scoring.contracts import Protocol, ScoreRequest, ScoreResponse, ServiceMetadata, WarmupRecord, digest_json


class ScoringClientError(RuntimeError):
    def __init__(self, message: str, *, fatal: bool = False, response: ScoreResponse | None = None):
        super().__init__(message)
        self.fatal = fatal
        self.response = response


def _require_wire_protocol(payload: dict) -> None:
    # Defaults help local constructors, but cannot attest to a remote backend.
    # Canonical JSON distinguishes omitted fields, 0 from false, and 8192.0 from 8192.
    try:
        actual = digest_json(payload.get("protocol"))
    except (TypeError, ValueError) as exc:
        raise ScoringClientError("invalid backend protocol", fatal=True) from exc
    if actual != digest_json(Protocol().model_dump()):
        raise ScoringClientError("backend must explicitly report the complete frozen protocol", fatal=True)


class ScoringClient:
    def __init__(self, base_url: str, timeout: float = 600):
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or not parts.netloc or parts.username or parts.password:
            raise ValueError("EVO2_BACKEND_URL must be an HTTP(S) URL without credentials")
        if parts.query or parts.fragment:
            raise ValueError("EVO2_BACKEND_URL must not contain a query or fragment")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._metadata: ServiceMetadata | None = None
        self._metadata_digest: str | None = None

    def _request(self, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = Request(self.base_url + path, data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.load(response)
        except HTTPError as exc:
            raise ScoringClientError(f"scoring backend HTTP {exc.code}", fatal=exc.code == 409) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ScoringClientError(f"scoring backend request failed: {type(exc).__name__}") from exc
        except (ValueError, UnicodeError) as exc:
            raise ScoringClientError("scoring backend returned invalid JSON", fatal=True) from exc
        if not isinstance(value, dict):
            raise ScoringClientError("scoring backend response must be an object", fatal=True)
        return value

    def metadata(self) -> ServiceMetadata:
        payload = self._request("/metadata")
        missing = ServiceMetadata.model_fields.keys() - payload.keys()
        if missing:
            raise ScoringClientError(f"backend metadata omitted fields: {', '.join(sorted(missing))}", fatal=True)
        _require_wire_protocol(payload)
        warmup = payload["warmup"]
        if not isinstance(warmup, dict) or WarmupRecord.model_fields.keys() - warmup.keys():
            raise ScoringClientError("backend must explicitly report the complete warmup record", fatal=True)
        try:
            observed = ServiceMetadata.model_validate(payload)
        except ValidationError:
            # Pydantic's rendered error includes rejected values, which may be labels.
            raise ScoringClientError("invalid scoring metadata: schema validation failed", fatal=True) from None
        if self._metadata_digest is not None and observed.sha256 != self._metadata_digest:
            raise ScoringClientError("backend metadata changed during run", fatal=True)
        self._metadata = observed
        self._metadata_digest = observed.sha256
        return observed

    def score(self, request: ScoreRequest) -> ScoreResponse:
        # Revalidate even model_construct/model_copy values before sending anything.
        request = ScoreRequest.model_validate(request.model_dump())
        metadata = self.metadata()
        payload = self._request("/score", request.model_dump())
        _require_wire_protocol(payload)
        try:
            result = ScoreResponse.model_validate(payload)
        except ValidationError:
            raise ScoringClientError("invalid scoring response: schema validation failed", fatal=True) from None
        if result.request != request or result.variant_id != request.variant_id:
            raise ScoringClientError("backend returned a different variant", fatal=True)
        if result.session_id != metadata.session_id or result.metadata_sha256 != self._metadata_digest:
            raise ScoringClientError("backend metadata or service session changed", fatal=True, response=result)
        if result.protocol != metadata.protocol:
            raise ScoringClientError("backend scoring protocol changed", fatal=True, response=result)
        if result.error_code == "metadata_drift":
            raise ScoringClientError("backend precision metadata changed", fatal=True, response=result)
        return result
