from pathlib import Path

filepath = Path("results/forest/qlearn")
filepath.parent.mkdir(parents=True, exist_ok=True)
filepath = Path("results/lake/qlearn")
filepath.parent.mkdir(parents=True, exist_ok=True)
