"""
bridge.py — Manages the coreai-runner subprocess lifecycle and proxies
HTTP requests over the Unix domain socket.

This is the ONLY Python file that touches the Swift binary. Nodes call
``bridge.predict(...)`` and never know there's a subprocess involved.

Lifecycle:
    1. First predict() call → ensure_running() → spawn subprocess
    2. Binary binds Unix socket, writes .ready file
    3. bridge waits for .ready (timeout 10s), then creates HTTP client
    4. Subsequent calls reuse the running process
    5. atexit → shutdown() → SIGTERM, cleanup socket
"""

from __future__ import annotations

import atexit
import logging
import os
import platform
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("ComfyUI-CoreAI")

# --- Constants -------------------------------------------------------------

SOCKET_PATH = "/tmp/coreai-runner.sock"
READY_TIMEOUT = 15.0
SHUTDOWN_TIMEOUT = 5.0
REQUEST_TIMEOUT = 300.0  # 5 min — large models (FLUX.2) can take 20s+

# GitHub Release binary URL (updated per release tag)
RUNNER_REPO = "kevinqz/coreai-runner"
RUNNER_VERSION = "1.0.0"  # matches GitHub Release tag


class CoreAIRunner:
    """Singleton managing the coreai-runner subprocess."""

    _instance: CoreAIRunner | None = None

    def __new__(cls) -> CoreAIRunner:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._process: subprocess.Popen | None = None
        self._client: httpx.Client | None = None
        self._binary_path: Path | None = None
        self._atexit_registered = False
        self._initialized = True

    # --- Public API --------------------------------------------------------

    def ensure_running(self) -> None:
        """Lazy-start the runner on first call. Idempotent."""
        if self._process and self._process.poll() is None:
            return  # already running

        self._binary_path = _resolve_binary()

        logger.info("Starting coreai-runner subprocess...")
        self._process = subprocess.Popen(
            [str(self._binary_path), "--socket", SOCKET_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "COREAI_LOG_LEVEL": "info"},
        )

        # Wait for the .ready file (signals the socket is listening)
        if not self._wait_for_ready():
            # Process died — capture stderr for diagnostics
            stderr = ""
            if self._process.stderr:
                stderr = self._process.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"coreai-runner failed to start within {READY_TIMEOUT}s.\n"
                f"stderr: {stderr[:500]}"
            )

        # Create HTTP client over Unix socket
        transport = httpx.HTTPTransport(uds=SOCKET_PATH)
        self._client = httpx.Client(
            transport=transport,
            base_url="http://unix",
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
        )
        logger.info("coreai-runner ready on %s", SOCKET_PATH)

        if not self._atexit_registered:
            atexit.register(self.shutdown)
            self._atexit_registered = True

    def shutdown(self) -> None:
        """Clean shutdown — SIGTERM, wait, SIGKILL if needed."""
        if self._client:
            self._client.close()
            self._client = None

        if self._process:
            if self._process.poll() is None:  # still running
                logger.info("Shutting down coreai-runner...")
                self._process.send_signal(signal.SIGTERM)
                try:
                    self._process.wait(timeout=SHUTDOWN_TIMEOUT)
                except subprocess.TimeoutExpired:
                    logger.warning("Runner didn't exit gracefully, killing...")
                    self._process.kill()
                    self._process.wait()
            self._process = None

        # Clean up socket + ready files
        for path in [SOCKET_PATH, SOCKET_PATH + ".ready"]:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def health(self) -> dict[str, Any]:
        """GET /v1/health — device info, loaded models."""
        self.ensure_running()
        assert self._client is not None
        resp = self._client.get("/v1/health")
        self._check_response(resp)
        return resp.json()

    def list_models(self, capability: str | None = None) -> list[dict[str, Any]]:
        """GET /v1/models — models filtered by capability."""
        self.ensure_running()
        assert self._client is not None
        params = {"capability": capability} if capability else {}
        resp = self._client.get("/v1/models", params=params)
        self._check_response(resp)
        return resp.json().get("models", [])

    def predict(
        self,
        model_id: str,
        image_path: str | None = None,
        prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        score_threshold: float | None = None,
        text_prompt: str | None = None,
        compute_unit: str = "auto",
    ) -> dict[str, Any]:
        """POST /v1/predict — run inference and return result.

        Wire keys inside ``input``/``options`` are snake_case — the SotA
        convention for this API, matching coreai-catalog, ComfyUI, and the
        HuggingFace/LLM ecosystem. The Swift runner maps these to its idiomatic
        camelCase Swift properties via explicit snake_case CodingKeys (see
        coreai-runner Codables.swift). Keep this in sync with that repo.
        """
        self.ensure_running()
        assert self._client is not None

        payload: dict[str, Any] = {
            "model_id": model_id,
            "input": {},
            "options": {"compute_unit": compute_unit},
        }
        if image_path:
            payload["input"]["image_path"] = image_path
        if prompt:
            payload["input"]["prompt"] = prompt
        if max_tokens is not None:
            payload["input"]["max_tokens"] = max_tokens
        if temperature is not None:
            payload["input"]["temperature"] = temperature
        if score_threshold is not None:
            payload["input"]["score_threshold"] = score_threshold
        if text_prompt:
            payload["input"]["text_prompt"] = text_prompt

        resp = self._client.post("/v1/predict", json=payload)
        self._check_response(resp, model_id=model_id)
        return resp.json()

    def load_model(self, model_id: str) -> dict[str, Any]:
        """POST /v1/models/:id/load — download + load model."""
        self.ensure_running()
        assert self._client is not None
        resp = self._client.post(f"/v1/models/{model_id}/load")
        self._check_response(resp, model_id=model_id)
        return resp.json()

    def unload_model(self, model_id: str) -> dict[str, Any]:
        """POST /v1/models/:id/unload — release model."""
        self.ensure_running()
        assert self._client is not None
        resp = self._client.post(f"/v1/models/{model_id}/unload")
        self._check_response(resp, model_id=model_id)
        return resp.json()

    def model_status(self, model_id: str) -> dict[str, Any]:
        """GET /v1/models/:id/status — installed, loaded, download progress."""
        self.ensure_running()
        assert self._client is not None
        resp = self._client.get(f"/v1/models/{model_id}/status")
        self._check_response(resp, model_id=model_id)
        return resp.json()

    def download_model(self, model_id: str) -> dict[str, Any]:
        """POST /v1/models/:id/load — trigger download without inference."""
        self.ensure_running()
        assert self._client is not None
        resp = self._client.post(f"/v1/models/{model_id}/load")
        self._check_response(resp, model_id=model_id)
        return resp.json()

    # --- Internal ----------------------------------------------------------

    @staticmethod
    def _check_response(resp, model_id: str | None = None) -> None:
        """Raise with the runner's error message instead of generic HTTP status."""
        if resp.is_success:
            return
        try:
            body = resp.json()
            err = body.get("error", body)
            code = err.get("code", "UNKNOWN")
            msg = err.get("message", resp.text[:200])
            mid = err.get("model_id", model_id)
            detail = f"[{code}] {msg}"
            if mid:
                detail = f"Model '{mid}': {detail}"
            raise RuntimeError(detail)
        except (ValueError, KeyError):
            resp.raise_for_status()

    def _wait_for_ready(self) -> bool:
        """Wait for the .ready file to appear (socket is listening).
        Verifies the PID inside the file matches our subprocess
        (detects stale .ready files from a previous crash)."""
        ready_path = SOCKET_PATH + ".ready"
        deadline = time.monotonic() + READY_TIMEOUT
        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                return False  # process died
            if os.path.exists(ready_path):
                # Verify the PID matches our process (stale file detection)
                try:
                    with open(ready_path, "r") as f:
                        ready_pid = int(f.read().strip())
                    if self._process and ready_pid == self._process.pid:
                        return True
                    # Stale .ready file (different PID) — remove and keep waiting
                    os.unlink(ready_path)
                except (ValueError, OSError):
                    # Can't read PID — just trust the file exists
                    return True
            time.sleep(0.05)
        return False


# --- Binary resolution -----------------------------------------------------

def _resolve_binary() -> Path:
    """
    Find the coreai-runner binary.
    Priority:
      1. COREAI_RUNNER_PATH env var (dev/override)
      2. Package-bundled binary
      3. Auto-download from GitHub Releases (arm64 macOS only)
    """
    # 1. Env override (dev mode)
    if env_path := os.environ.get("COREAI_RUNNER_PATH"):
        p = Path(env_path)
        if p.exists():
            return p
        logger.warning("COREAI_RUNNER_PATH=%s does not exist", env_path)

    # 2. Bundled
    pkg_dir = Path(__file__).parent / "bin"
    binary = pkg_dir / "coreai-runner"
    if binary.exists() and os.access(binary, os.X_OK):
        return binary

    # 3. Auto-download
    return _download_binary()


def _download_binary() -> Path:
    """Download the pre-compiled binary from GitHub Releases."""
    arch = platform.machine()
    os_name = platform.system()

    if arch != "arm64" or os_name != "Darwin":
        raise RuntimeError(
            f"ComfyUI-CoreAI requires Apple Silicon (arm64 macOS). "
            f"Detected: {arch}/{os_name}"
        )

    pkg_dir = Path(__file__).parent / "bin"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    binary = pkg_dir / "coreai-runner"

    if binary.exists() and os.access(binary, os.X_OK):
        return binary  # already downloaded

    url = (
        f"https://github.com/{RUNNER_REPO}/releases/download/"
        f"v{RUNNER_VERSION}/coreai-runner-arm64-macos"
    )
    logger.info("Downloading coreai-runner binary from %s", url)

    # Stream download with progress
    import urllib.request

    urllib.request.urlretrieve(url, binary)
    os.chmod(binary, 0o755)

    # Verify checksum if available
    checksum_url = url + ".sha256"
    try:
        import hashlib

        resp = urllib.request.urlopen(checksum_url, timeout=10)
        expected = resp.read().decode().strip().split()[0]
        actual = hashlib.sha256(binary.read_bytes()).hexdigest()
        if expected != actual:
            binary.unlink()
            raise RuntimeError(
                f"Binary checksum mismatch: expected {expected}, got {actual}"
            )
        logger.info("Binary checksum verified")
    except Exception as e:
        logger.warning("Could not verify checksum: %s", e)

    return binary


# --- Singleton accessor ----------------------------------------------------

def get_runner() -> CoreAIRunner:
    """Get the singleton runner instance."""
    return CoreAIRunner()
