# Windmills at Sea (Modelling 4) — beam prototype

Local project for TU Delft TW2-41: cantilever beam as a wind tower, semi-analytical modal model.

- `docs/WindProject2.pdf` — assignment brief
- `dynamic_beam_model.py` — minimal semi-analytical modal model (tip harmonic load)
- `first_beam_model.py` — **homogeneous beam only**: cantilever eigenvalues / natural frequencies, normalized mode shapes `phi_n(x)`, plot saved as `eigenmodes.png` (gitignored)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python dynamic_beam_model.py
python first_beam_model.py
```
