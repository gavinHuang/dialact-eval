"""
conftest.py — Add dialact-eval root to sys.path so that core/, eval/, ui/
are importable as top-level packages during pytest runs.
"""
import sys
from pathlib import Path

# Insert the project root so `import core`, `import eval`, `import ui` resolve.
root = Path(__file__).parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
