"""
Runtime helpers for local Squad-style setup, display, and file materialization.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SQUAD_DIR = Path(".squad")
SQUAD_CONFIG = SQUAD_DIR / "config.json"
LOG_DIR = SQUAD_DIR / "log"
WORK_DIR = SQUAD_DIR / "work"


@dataclass
class FileWrite:
    """A file path and exact content proposed by an agent."""

    path: str
    content: str


def init_squad(project_root: str = ".") -> list[str]:
    """Create the local .squad runtime folders and default config."""
    created: list[str] = []
    project_path = Path(project_root).expanduser()
    if not project_path.is_absolute():
        project_path = (Path.cwd() / project_path).resolve()
    else:
        project_path = project_path.resolve()

    for directory in (SQUAD_DIR, LOG_DIR, WORK_DIR):
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(str(directory))

    config = {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "project_root": str(project_path),
        "log_dir": str(LOG_DIR),
        "work_dir": str(WORK_DIR),
        "file_write_format": "FILE: relative/path followed by a fenced code block",
        "agents": {
            "planner": {"role": "planning"},
            "coder": {"role": "coding", "can_write_files": True},
            "fast_utility": {"role": "utility", "can_write_files": False},
        },
    }

    if not SQUAD_CONFIG.exists():
        SQUAD_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")
        created.append(str(SQUAD_CONFIG))

    return created


def get_project_root() -> Path:
    """Return the configured project root, defaulting to the current directory."""
    if SQUAD_CONFIG.exists():
        try:
            config = json.loads(SQUAD_CONFIG.read_text(encoding="utf-8"))
            configured_root = config.get("project_root")
            if configured_root:
                return Path(configured_root).expanduser().resolve()
        except (OSError, json.JSONDecodeError):
            pass

    return Path.cwd().resolve()


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def _try_load_json(text: str) -> Any | None:
    candidates = [_strip_code_fence(text)]
    fenced_json = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates.extend(chunk.strip() for chunk in fenced_json)
    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start != -1 and object_end > object_start:
        candidates.append(text[object_start : object_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def format_plan_for_display(plan: str) -> str:
    """Turn the planner's JSON into readable terminal text when possible."""
    data = _try_load_json(plan)
    if not isinstance(data, dict):
        return plan

    lines: list[str] = []
    title = data.get("task_title")
    complexity = data.get("complexity")
    if title:
        prefix = f"{title}"
        if complexity:
            prefix += f" ({complexity})"
        lines.append(prefix)

    steps = data.get("steps")
    if isinstance(steps, list):
        for item in steps:
            if not isinstance(item, dict):
                continue
            step = item.get("step", len(lines) + 1)
            action = item.get("action", "")
            agent = item.get("assigned_agent")
            notes = item.get("notes")
            line = f"{step}. {action}".strip()
            if agent:
                line += f" [{agent}]"
            if notes:
                line += f" - {notes}"
            lines.append(line)

    notes = data.get("notes")
    if notes:
        lines.append(f"Notes: {notes}")

    return "\n".join(lines) if lines else plan


def extract_file_writes(response: str) -> list[FileWrite]:
    """
    Extract proposed file writes from common model formats.

    Supported formats include:
    - {"files": [{"path": "src/app.py", "content": "..."}]}
    - FILE: src/app.py followed by a fenced code block
    - Markdown heading containing a path followed by a fenced code block
    """
    files: list[FileWrite] = []
    data = _try_load_json(response)

    if isinstance(data, dict):
        json_files = data.get("files") or data.get("file_writes") or data.get("changes")
        if isinstance(json_files, list):
            for item in json_files:
                if not isinstance(item, dict):
                    continue
                path = item.get("path") or item.get("file") or item.get("filename")
                content = item.get("content") or item.get("code")
                if isinstance(path, str) and isinstance(content, str):
                    files.append(FileWrite(path=path.strip(), content=content))

    labelled_pattern = re.compile(
        r"(?:^|\n)(?:#{1,6}\s*)?(?:FILE|File|file|Path|path|Filename|filename):\s*`?([^\n`]+?)`?\s*\n"
        r"```[^\n]*\n(.*?)\n```",
        re.DOTALL,
    )
    for path, content in labelled_pattern.findall(response):
        files.append(FileWrite(path=path.strip(), content=content))

    heading_pattern = re.compile(
        r"(?:^|\n)#{1,6}\s+`?([A-Za-z0-9_./\\-]+\.[A-Za-z0-9_+-]+)`?\s*\n"
        r"```[^\n]*\n(.*?)\n```",
        re.DOTALL,
    )
    for path, content in heading_pattern.findall(response):
        files.append(FileWrite(path=path.strip(), content=content))

    info_string_pattern = re.compile(
        r"```[A-Za-z0-9_+.-]*\s+(?:path|file|filename)=['\"]?([^'\"\n]+?)['\"]?\s*\n(.*?)\n```",
        re.DOTALL,
    )
    for path, content in info_string_pattern.findall(response):
        files.append(FileWrite(path=path.strip(), content=content))

    deduped: list[FileWrite] = []
    seen: set[str] = set()
    for file_write in files:
        key = file_write.path.replace("\\", "/")
        if key and key not in seen:
            seen.add(key)
            deduped.append(file_write)

    return deduped


def _resolve_safe_path(project_root: Path, file_path: str) -> Path:
    raw_path = Path(file_path.strip().strip("'\""))
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    else:
        resolved = (project_root / raw_path).resolve()

    if resolved != project_root and project_root not in resolved.parents:
        raise ValueError(f"Refusing to write outside project root: {file_path}")

    return resolved


def write_agent_files(files: list[FileWrite], project_root: Path | None = None) -> list[str]:
    """Write extracted file proposals into the configured project root."""
    root = (project_root or get_project_root()).resolve()
    written: list[str] = []

    for file_write in files:
        target = _resolve_safe_path(root, file_write.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file_write.content, encoding="utf-8")
        written.append(str(target))

    return written
