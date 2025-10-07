"""Root conftest.py for pytest configuration."""
import sys
from pathlib import Path

# Ensure project root is in Python path BEFORE any test imports
# This must run first so that test modules can import tools.*
project_root = Path(__file__).parent.resolve()
project_root_str = str(project_root)

# Insert at position 0 to ensure it's checked first
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# Verify tools package is importable
try:
    import tools.backup
except ImportError as e:
    print(f"WARNING: Cannot import tools.backup: {e}")
    print(f"Project root: {project_root}")
    print(f"sys.path: {sys.path[:5]}")
