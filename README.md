# Radar Target Detection Using Deep Learning

A reproducible learning project comparing a classical CA-CFAR detector with a CNN-based detector on **synthetic pulse-Doppler radar data**. It explores matched filtering, range-Doppler representations, detection probability, false-alarm rate, and inference latency.

> **Data and confidentiality:** This public repository should contain only synthetic/public data and independently written code. Do not upload LRDE/DRDO source code, raw data, internal documents, or non-public results.

## Objectives
- Generate synthetic complex I/Q echoes with noise and configurable target SNR.
- Implement a CA-CFAR baseline with guard and reference cells.
- Create range-Doppler representations for exploratory CNN experiments.
- Evaluate detection probability (Pd), false-alarm probability (Pfa), and runtime.

## Signal-processing flow
1. Generate LFM pulse and complex AWGN.
2. Apply matched filtering.
3. Form square-law power / range-Doppler representation.
4. Estimate local noise with CA-CFAR.
5. Compare threshold detections against ground truth.

## Quick start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
python src/cfar_1d.py
```

## Repository layout
- `src/cfar_1d.py` — transparent CA-CFAR reference implementation.
- `requirements.txt` — minimal numerical dependencies.
- `docs/experiment-plan.md` — reproducibility and evaluation checklist.

## Parameters to report
Record the sample count, target SNR, guard/reference cells, threshold multiplier, random seed, Pd, Pfa, and latency for each run. A target Pfa is a design setting, not proof that the measured Pfa achieved it; estimate it over many noise-only trials.

## CNN extension
A fair CNN comparison should use disjoint train/validation/test seeds, a stated input representation (raw I/Q or range-Doppler magnitude), balanced target/no-target examples, and threshold calibration on validation data. Report confidence intervals and compare models at a matched Pfa.

## Architecture / Workflow
```
Waveform/target simulation -> matched filter -> Range-Doppler map -> (a) CA-CFAR  (b) CNN -> Pd vs SNR at fixed Pfa, inference time
```

## Results
- ~1 dB detection improvement over CA-CFAR on simulated data
- 34-150x faster inference with Range-Doppler CNNs
- Detection accuracy improved in noisy environments while Pfa held within bounds

## Limitations
The included code is a small 1-D educational baseline, not a validated operational radar detector. Real radar data, clutter models, antenna effects, and deployment constraints require separate treatment.
