# run_app.py
from pathlib import Path
import sys

# Ensure repo root is on sys.path no matter where Streamlit runs from
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Hand off to the actual Streamlit script
# import runpy
# runpy.run_path(str(ROOT / "rag_app" / "ui" / "app.py"), run_name="__main__")
print("Streamlit UI has been migrated to Chainlit.")
print("Please run: chainlit run rag_app/ui/chat.py")
