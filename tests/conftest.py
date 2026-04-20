import sys
from pathlib import Path

# Add src/ to path so tests can import `quantum.*`, `core.*`, `data.*` directly
src = str(Path(__file__).resolve().parent.parent / 'src')
if src not in sys.path:
    sys.path.insert(0, src)

from core.utils import set_seed  # noqa: E402

set_seed(42)
