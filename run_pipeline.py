import importlib.util
import subprocess
import sys
from typing import List, Tuple


def ensure_packages(packages: List[Tuple[str, str]]) -> None:
	"""Ensure each (import_name, pip_name) is importable in this interpreter.

	If missing, install with `python -m pip install <pip_name>`.
	"""
	for import_name, pip_name in packages:
		if importlib.util.find_spec(import_name) is None:
			print(f"Package '{import_name}' not found — installing '{pip_name}' into {sys.executable}...")
			subprocess.run([sys.executable, "-m", "pip", "install", pip_name], check=True)


def main() -> None:
	print("🚀 Starting automated real-time weather pipeline")

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
		print(f"Pipeline step failed: {exc}")
		raise

	print("✅ Pipeline finished successfully")


if __name__ == "__main__":
	main()
