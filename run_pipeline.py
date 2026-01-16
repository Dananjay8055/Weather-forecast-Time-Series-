import importlib.util
import subprocess
import sys
from typing import List, Tuple


# Try to ensure stdout/stderr use UTF-8 on Windows consoles. If reconfigure
# isn't available (older Python or non-writable streams), fall back to a
# safe print helper that replaces non-encodable characters.
try:
	sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
	sys.stderr.reconfigure(encoding='utf-8')  # type: ignore
except Exception:
	pass


def _safe_print(msg: str) -> None:
	try:
		print(msg)
	except UnicodeEncodeError:
		enc = getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8'
		safe = msg.encode(enc, errors='replace').decode(enc)
		print(safe)


def ensure_packages(packages: List[Tuple[str, str]]) -> None:
	"""Ensure each (import_name, pip_name) is importable in this interpreter.

	If missing, install with `python -m pip install <pip_name>`.
	"""
	for import_name, pip_name in packages:
		if importlib.util.find_spec(import_name) is None:
			_safe_print(f"Package '{import_name}' not found — installing '{pip_name}' into {sys.executable}...")
			subprocess.run([sys.executable, "-m", "pip", "install", pip_name], check=True)


def main() -> None:
	_safe_print("🚀 Starting automated real-time weather pipeline")

	# Ensure runtime packages are available in the interpreter running the pipeline
	ensure_packages([
		("requests", "requests"),
		("pandas", "pandas"),
		("sklearn", "scikit-learn"),
		("matplotlib", "matplotlib"),
		("joblib", "joblib"),
	])

	try:
		subprocess.run([sys.executable, "fetch_noaa_data.py"], check=True)
		subprocess.run([sys.executable, "train_model.py"], check=True)
	except subprocess.CalledProcessError as exc:
		_safe_print(f"Pipeline step failed: {exc}")
		raise

	_safe_print("✅ Pipeline finished successfully")


if __name__ == "__main__":
	main()
