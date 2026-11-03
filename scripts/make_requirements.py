"""Write requirements.txt using the library versions recorded when the model was saved."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
v = json.loads((ROOT / "models" / "model_card.json").read_text())["versions"]

pkgs = ["lightgbm", "scikit-learn", "pandas", "numpy", "joblib", "pyarrow", "plotly"]
lines = ["streamlit"] + [f"{p}=={v[p]}" for p in pkgs if v.get(p)]
(ROOT / "requirements.txt").write_text("\n".join(lines) + "\n")

print("\n".join(lines))
print("\nModel was trained with Python", v.get("python"))