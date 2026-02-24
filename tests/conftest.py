from __future__ import annotations

import sys
from pathlib import Path


# Ensure the repository root is on sys.path so `import src.*` works
# regardless of the current working directory used to invoke pytest.
REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)
