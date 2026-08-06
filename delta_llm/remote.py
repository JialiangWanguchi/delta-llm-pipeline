from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

RESULT_PREFIX = "DELTA_LLM_RESULT|"


@dataclass(frozen=True)
class RemoteResult:
    deployment_id: str
    job_id: str
    state: str
    endpoint: str
    expires_at: str
    remote_dir: str


def validate_username(username: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", username):
        raise ValueError("NCSA username contains unsupported characters")
    return username


def validate_deployment_id(deployment_id: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{5,63}", deployment_id):
        raise ValueError("Invalid deployment ID")
    return deployment_id


class SSHRunner:
    def __init__(self, username: str, host: str) -> None:
        self.username = validate_username(username)
        self.host = host

    @property
    def destination(self) -> str:
        return f"{self.username}@{self.host}"

    def run_script(self, script: str) -> RemoteResult | None:
        """Run one remote script through one password+Duo-authenticated SSH session.

        OpenSSH reads authentication prompts from the controlling terminal while
        stdin carries the script. stdout is mirrored live and a structured marker
        is parsed when present.
        """

        command = [
            "ssh",
            "-T",
            "-o",
            "ControlMaster=no",
            "-o",
            "ControlPath=none",
            "-o",
            "PreferredAuthentications=keyboard-interactive,password",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=6",
            self.destination,
            "bash -s",
        ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("OpenSSH client not found. Install/enable the 'ssh' command.") from exc

        assert process.stdin is not None
        assert process.stdout is not None
        wire_script = script.replace("\r\n", "\n").replace("\r", "\n")
        process.stdin.write(wire_script.encode("utf-8"))
        process.stdin.close()

        parsed: RemoteResult | None = None
        for raw_line in process.stdout:
            line = raw_line.decode("utf-8", errors="replace")
            print(line, end="", flush=True)
            if line.startswith(RESULT_PREFIX):
                parsed = parse_result_line(line.rstrip("\r\n"))

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Remote SSH operation failed with exit code {return_code}")
        return parsed


def parse_result_line(line: str) -> RemoteResult:
    parts = line.split("|", 6)
    if len(parts) != 7 or parts[0] != "DELTA_LLM_RESULT":
        raise ValueError(f"Malformed result marker: {line}")
    return RemoteResult(
        deployment_id=parts[1],
        job_id=parts[2],
        state=parts[3],
        endpoint=parts[4],
        expires_at=parts[5],
        remote_dir=parts[6],
    )


def local_state_root() -> Path:
    override = os.environ.get("DELTA_LLM_STATE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".delta-llm" / "deployments"


def save_local_state(deployment_id: str, payload: str) -> Path:
    root = local_state_root()
    root.mkdir(parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    path = root / f"{deployment_id}.json"
    path.write_text(payload, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def ensure_interactive_terminal() -> None:
    if not sys.stdin.isatty():
        raise RuntimeError(
            "This command needs an interactive terminal for the NCSA password and Duo prompt."
        )
