from __future__ import annotations

from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else project_root() / "config.yaml"
    try:
        import yaml
    except ImportError:
        return _minimal_yaml_load(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root() / path


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _minimal_yaml_load(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            item = _parse_scalar(line[2:])
            if isinstance(parent, list):
                parent.append(item)
            continue

        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
            continue

        next_container: Any = {}
        parent[key] = next_container
        stack.append((indent, next_container))

    # Convert mapping placeholders that were intended as lists.
    text = path.read_text(encoding="utf-8").splitlines()
    for idx, raw_line in enumerate(text):
        stripped = raw_line.strip()
        if not stripped.endswith(":"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        following = next((line for line in text[idx + 1 :] if line.strip()), "")
        if following.strip().startswith("- ") and len(following) - len(following.lstrip(" ")) > indent:
            keys = [part.strip() for part in stripped[:-1].split(".")]
            # The current config only needs one-level list conversions.
            for parent_key, parent_value in root.items():
                if isinstance(parent_value, dict) and keys[-1] in parent_value:
                    list_indent = len(following) - len(following.lstrip(" "))
                    items = []
                    for item_line in text[idx + 1 :]:
                        if not item_line.strip():
                            continue
                        item_indent = len(item_line) - len(item_line.lstrip(" "))
                        if item_indent < list_indent:
                            break
                        if item_line.strip().startswith("- "):
                            items.append(_parse_scalar(item_line.strip()[2:]))
                    parent_value[keys[-1]] = items
    return root
