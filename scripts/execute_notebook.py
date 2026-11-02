"""Execute the CPU notebook without suppressing failures."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root = Path(__file__).resolve().parents[1]
path = root / "notebooks/01_data_and_baseline.ipynb"
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
nbformat.write(nb, path)
print(f"Executed {path.name} successfully")
