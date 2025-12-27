# %% ------------------------------------------------------------
# Libraries
# ---------------------------------------------------------------
%matplotlib qt
import json
from pathlib import Path
from typing import Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import jax.nn as jnn
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from jax import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # for saving scalers
import seaborn as sns
import itertools
import datetime
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------
# Config / Columns
# ---------------------------------------------------------------
COLUMNS = {
    "Rock": [
        "agitation_rate (cpm)",
        "base_height (mm)",
        "start_angle (deg)",
        "stop_angle (deg)",
        "up_dwell",
        "down_dwell",
        "split_period",
        "bio_diameter (mm)",
        "fluid_volume",
        "fluid_viscosity (mPas)",
        "fluid_density (kg/m^3)",
    ],
    "Compression": [
        "agitation_Rate (cpm)",
        "stroke_length (mm)",
        "base_height (mm)",
        "up_dwell (s)",
        "down_dwell (s)",
        "split_period",
        "baffle_diameter (mm)",
        "bio_diameter (mm)",
        "fluid_volume (ml)",
        "MCV_of_single_cell (ml)",
        "number_cells",
        "cell_vol/fluid_vol",
        "fluid_viscosity (mPas)",
        "fluid_density (kg/m^3)",
    ],
}

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def inclusive_range(start: float, step: float, end: float):
    """Return start:step:end (inclusive). Handles ascending or descending."""
    if step == 0:
        return [start]
    out = []
    if start <= end:
        x = start
        while x <= end + 1e-12:
            out.append(round(x, 10))
            x += step
    else:
        x = start
        while x >= end - 1e-12:
            out.append(round(x, 10))
            x -= step
    return out

def build_base_design():
    # ---------- Base factor levels ----------
    mixing_type = "Rock"
    fluid_volume_ml = [250, 500]  # ml
    Dye_volume = fluid_volume_ml[0] * 0.48  # ul
    Injection_time = 27
    flow_rate = Dye_volume * 1e-3 / Injection_time * 60  # ul/s

    # 10..60 cpm (step 1)
    agitation_rate_cpm = range(5, 61, 2)

    up_dwell_s = inclusive_range(0, 0, 0)
    down_dwell_s = inclusive_range(0, 0, 0)
    start_angle_deg = inclusive_range(-10, 0, -10)
    down_angle_deg = inclusive_range(10, 0, 10)
    base_height_mm = inclusive_range(0, 0, 0)
    split_period_s = inclusive_range(0.5, 0, 0.5)

    # Constants (ensure units match what your model was trained on)
    base_diameter_mm = 140
    fluid_viscosity_mPas = 1.0e-3        # if training used mPa·s ~ water ≈ 1
    fluid_density_kg_m3 = 1000

    # ---------- Build base design ----------
    iterables = (
        fluid_volume_ml,
        agitation_rate_cpm,
        up_dwell_s,
        down_dwell_s,
        start_angle_deg,
        down_angle_deg,
        base_height_mm,
        split_period_s,
    )

    combos = list(itertools.product(*iterables))
    design = pd.DataFrame(
        combos,
        columns=[
            "fluid_volume",
            "agitation_rate (cpm)",
            "up_dwell",
            "down_dwell",
            "start_angle (deg)",
            "stop_angle (deg)",
            "base_height (mm)",
            "split_period",
        ],
    )

    # Add fixed columns and metadata
    design.insert(1, "mix_type", mixing_type)
    design["bio_diameter (mm)"] = base_diameter_mm
    design["fluid_viscosity (mPas)"] = fluid_viscosity_mPas
    design["fluid_density (kg/m^3)"] = fluid_density_kg_m3
    design["flow_rate_ul_s"] = round(flow_rate, 2)
    design["Dye_volume_ul"] = Dye_volume
    design["Injection_time_s"] = Injection_time
    design["date_created"] = datetime.datetime.now().strftime("%d-%m-%Y")
    design["created_by"] = "Weheliye"
    design["pred"] = None
    return design

def numpy_to_jax(pytree):
    # Back to jnp arrays on load
    return jax.tree.map(lambda a: jnp.array(a), pytree)

def load_bundle(load_path: Path):
    bundle = joblib.load(load_path)
    bundle["params"] = numpy_to_jax(bundle["params"])
    print(f"Loaded model bundle from: {load_path}")
    return bundle

# ---------------------------------------------------------------
# Model (Haiku) with Dropout (used at train-time; inference disables it)
# ---------------------------------------------------------------
class Detector(hk.Module):
    """
    Simple MLP with ReLU + dropout after each hidden layer.
    """
    def __init__(self, layer_sizes: Sequence[int], activation: str, dropout_rate: float = 0.2, name=None):
        super().__init__(name=name)
        self.layer_sizes = layer_sizes
        self.dropout_rate = float(dropout_rate)
        self.activation = activation

    def __call__(self, x, is_training: bool):
        # Hidden layers
        for i, size in enumerate(self.layer_sizes[:-1]):
            x = hk.Linear(size, name=f"linear_{i}")(x)
            if self.activation == "relu":
                x = jnn.relu(x)
              #  print('Using ReLU activation')
            if self.dropout_rate > 0.0 and is_training:
                x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        # Output layer (no activation for regression)
        out_size = self.layer_sizes[-1]
        return hk.Linear(out_size, name="output")(x)


# ---------------------------------------------------------------
# Inference utility
# ---------------------------------------------------------------
def process_data(Folder_type: str, Exp_type: str, X_val_df: pd.DataFrame):
    """
    Load the trained bundle from `Folder_type/model/model_{Exp_type}.joblib`
    and run inference (dropout OFF).
    """
    model_dir = Path(Folder_type).joinpath("model")
    model_path = model_dir.joinpath(f"model_{Exp_type}.joblib")
    print("Loading:", model_path)

    bundle = load_bundle(model_path)
    params = bundle["params"]
    x_scaler = bundle["x_scaler"]
    cfg = bundle["config"]

    # Scale inputs with the saved scaler
    X_scaled = x_scaler.transform(X_val_df.values).astype(np.float32)

    # Rebuild the network shape (dropout rate read from config; disabled at eval)
    def forward_fn(x, is_training: bool):
        #net = Detector(cfg["architecture"], dropout_rate=cfg.get("dropout_rate", 0.0))
        net = Detector(cfg["architecture"], activation=cfg['activation'],dropout_rate=cfg['dropout_rate'])
        return net(x, is_training)

    model = hk.transform(forward_fn)

    # Inference (eval mode => no dropout). RNG can be any fixed key.
    rng = random.PRNGKey(0)
    preds = model.apply(params, rng, jnp.array(X_scaled), False)

    return np.asarray(preds).squeeze(-1)

# ---------------------------------------------------------------
# Build inference grid and run models
# ---------------------------------------------------------------
data = build_base_design()

FOLDER = ["Mixing_time", "KLa"]     # the two trained models to compare
EXP_TYPE = "Rock"

X_val_df = data[COLUMNS[EXP_TYPE]].copy()   # for plotting and enrichment
X_val    = data[COLUMNS[EXP_TYPE]].copy()   # features fed into the models

for folder in FOLDER:
    X_val_df[folder] = process_data(Folder_type=folder, Exp_type=EXP_TYPE, X_val_df=X_val)
#%%
# Simple label based on agitation rate
X_val_df["Solid Suspension"] = X_val_df["Mixing_time"].apply(
    lambda x: "aggregated" if x <= 210 else "Fully suspended"
)

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=X_val_df,
    x=FOLDER[1],   # Mixing_time
    y=FOLDER[0],   # KLa
    hue="fluid_volume",
    style="Solid Suspension",
    markers=["^", "o"],
    palette=["blue", "red"],
    s=100
)

# Add green rectangle (example ROI)
#rect = Rectangle((150, 0), 62, 105, facecolor="green", alpha=0.4)
rect = Rectangle((0, 150), 105, 62, facecolor="green", alpha=0.4)
plt.gca().add_patch(rect)

plt.text(
    50, 300, "Good Growth\nZone",
    bbox=dict(facecolor='white', alpha=0.2, edgecolor="none"),
    ha="left", va="center"
)
plt.xlim(left=0, right=80)
plt.yscale("log")
plt.xlabel("KLa (1/s)")
plt.ylabel("Mixing time (s)")
plt.tight_layout()
plt.savefig(f"Mixing_time_vs_KLa_{EXP_TYPE}_two_fluid_volume.png", dpi=300)
plt.show()

# %%
