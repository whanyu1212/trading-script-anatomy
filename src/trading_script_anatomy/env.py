"""Environment-file loading utilities."""

import os
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines into the environment without overriding it.

    Existing environment variables win over file values, and missing files
    are ignored. Call sites opt in explicitly; library code never loads
    ``.env`` files implicitly.

    Args:
        path: Dotenv file location.
    """
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
