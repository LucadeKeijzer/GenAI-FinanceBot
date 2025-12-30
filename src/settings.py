import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

DEFAULT_SETTINGS_PATH = Path("user_settings.json")


@dataclass
class UserSettings:
    experience_level: str = "Beginner"   # Beginner / Intermediate / Advanced
    budget_range: str = "€100-1000"      # €0–100 / €100–1000 / €1000+
    detail_level: str = "Simple"         # Simple / Advanced
    language: str = "English"            # English (Dutch later)


def load_user_settings(path: Path = DEFAULT_SETTINGS_PATH) -> Optional[UserSettings]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return UserSettings(
            experience_level=str(data.get("experience_level", "Beginner")),
            budget_range=str(data.get("budget_range", "€100–1000")),
            detail_level=str(data.get("detail_level", "Simple")),
            language=str(data.get("language", "English")),
        )
    except Exception:
        # Corrupted JSON => treat as first run
        return None


def save_user_settings(settings: UserSettings, path: Path = DEFAULT_SETTINGS_PATH) -> None:
    path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")