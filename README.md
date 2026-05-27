# HW1-2 LQR Vehicle Lateral Control

This submission implements and analyzes an LQR controller for vehicle lateral path tracking.

## Contents

- `HW1-2_LQR.ipynb` - main notebook with derivation, simulations, plots, and tuning observations.
- `LQR_Full_state_first_step.py` - reusable Python implementation for the vehicle model, LQR controller, path generation, simulation, metrics, and plots.
- `HW1-2_LQR.pdf` - rendered PDF version of the notebook with math formulas displayed.
- `HW1-2_LQR_math.html` - HTML version used to render equations correctly with MathJax.
- `objective2_outputs/` - saved plot images.
- `requirements.txt` - Python dependencies.
- `HW1-2_LQR_submission.zip` - packaged submission archive.

## Objectives Covered

1. Derive the lateral dynamic bicycle state-space model.
2. Implement LQR path tracking on multiple reference paths.
3. Tune `Q` and `R` to compare aggressive tracking, smooth steering, reduced lateral error, and reduced heading error.

## Setup

Create or activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Notebook

Open and run:

```bash
jupyter notebook HW1-2_LQR.ipynb
```

or open the notebook in VS Code and run all cells.

The optional interactive cell uses `ipywidgets`. If the widget controls do not display, restart the notebook kernel after installing dependencies.

## Run the Script

The standalone script can also be run directly:

```bash
python LQR_Full_state_first_step.py
```

It prints LQR gains, tracking metrics, and displays the generated plots.

## Notes

The controller uses a four-state LQR model:

```text
x = [e, psi_e, v_y, r]^T
```

where `e` is cross-track error, `psi_e` is heading error, `v_y` is lateral velocity, and `r` is yaw rate. The implementation includes curvature feedforward steering and steering saturation.
