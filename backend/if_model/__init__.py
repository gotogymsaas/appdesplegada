import sys
from pathlib import Path

# Ruta del archivo feature_engineer_v6.py
current_dir = Path(__file__).resolve().parent
module_file = current_dir / "feature_engineer_v6.py"

# Agregar a path del sistema
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
