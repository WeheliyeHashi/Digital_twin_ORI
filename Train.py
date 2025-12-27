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
        #"MCV_of_single_cell (ml)",
        #"number_cells",
        #"cell_vol/fluid_vol",
        "fluid_viscosity (mPas)",
        "fluid_density (kg/m^3)",
    ],
}

# Choose which folder and experiment type to use
FOLDER = "Mixing_time"   # e.g. "KLa", "Mixing_time", "Shear_stress"
DATA_PATH = r"Mixing_time\model_2\mixing_data.xlsx"
EXP_TYPE = "Rock"        # "Rock" or "Compression"

# ---------------------------------------------------------------
# Model path creation and hyperparameters
# ---------------------------------------------------------------
model_dir = Path(FOLDER).joinpath("model_2")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir.joinpath(f"model_{EXP_TYPE}.joblib")

Detector_architecture = [128, 128,64,64, 1]  # Example architecture
DROPOUT_RATE = 0.1  # Try 0.1–0.3
activation  ="tanh" # Try "relu" or "linear"
csv_path = Path(DATA_PATH)
data = pd.read_excel(csv_path, index_col=False)
data['y'] = data["Mixing time [s]"]

# ---------------------------------------------------------------
# Save / Load utilities
# ---------------------------------------------------------------
def params_to_numpy(pytree):
    # Move from device to host + make plain numpy arrays for safe pickling
    return jax.tree.map(lambda a: np.asarray(jax.device_get(a)), pytree)

def numpy_to_jax(pytree):
    # Back to jnp arrays on load
    return jax.tree.map(lambda a: jnp.array(a), pytree)

def save_bundle(save_path: Path, params, x_scaler, extra_cfg: dict):
    bundle = {
        "params": params_to_numpy(params),
        "x_scaler": x_scaler,     # sklearn objects pickle well with joblib
        "config": extra_cfg,      # any metadata you want to carry along
    }
    joblib.dump(bundle, save_path)
    print(f"Saved model bundle to: {save_path}")

def load_bundle(load_path: Path):
    bundle = joblib.load(load_path)
    bundle["params"] = numpy_to_jax(bundle["params"])
    print(f"Loaded model bundle from: {load_path}")
    return bundle

# ---------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------
def build_search_spaces(exp_data: pd.DataFrame, exp_type="Rock"):
    """Build (min, max) ranges for each column, save for reference."""
    bounds = {col: (exp_data[col].min(), exp_data[col].max()) for col in exp_data.columns}
    return {exp_type: bounds}

def create_train_test(data: pd.DataFrame, exp_type: str, folder: str):
    #exp_data = data[data["mix_type"] == exp_type].copy()
    exp_data = data.copy()
    X = exp_data[COLUMNS[exp_type]].copy()
    y = exp_data["y"].copy()

    # Save search space (nice for later hyperparameter sampling)
    search_space = build_search_spaces(exp_data[X.columns.tolist() + ["y"]], exp_type)

    def convert_np(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(Path(folder).joinpath(f"search_space_{folder}.json"), "w") as f:
        json.dump(search_space, f, indent=4, default=convert_np)

    return train_test_split(X, y, test_size=0.5, random_state=42)

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------


X_train_df, X_test_df, y_train_sr, y_test_sr = create_train_test(data, EXP_TYPE, FOLDER)

# ---------------------------------------------------------------
# Normalization (X and y)
# ---------------------------------------------------------------
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train_df.values).astype(np.float32)
X_test  = x_scaler.transform(X_test_df.values).astype(np.float32)

y_train = y_train_sr.values.reshape(-1, 1).astype(np.float32)
y_test  = y_test_sr.values.reshape(-1, 1).astype(np.float32)

# Pre-create JAX arrays once (avoid per-batch conversions)
X_train_jnp = jnp.array(X_train)
y_train_jnp = jnp.array(y_train)
X_test_jnp  = jnp.array(X_test)
y_test_jnp  = jnp.array(y_test)
print(f'hyperparamters: activation {activation}, dropout {DROPOUT_RATE}, arch {Detector_architecture}')

# ---------------------------------------------------------------
# Model (Haiku) with Dropout
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
               # print('Using ReLU activation')
            if self.activation == "tanh":
                x = jnn.mish(x)
               # print('using tanh')
            if self.dropout_rate > 0.0 and is_training:
                x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        # Output layer (no activation for regression)
        out_size = self.layer_sizes[-1]
        return hk.Linear(out_size, name="output")(x)

def forward_fn(x, is_training: bool):
    model = Detector(Detector_architecture, activation = activation, dropout_rate=DROPOUT_RATE)
    return model(x, is_training)

model = hk.transform(forward_fn)
rng = random.PRNGKey(42)
#%%
# ---------------------------------------------------------------
# Init
# ---------------------------------------------------------------

input_example = X_train_jnp[:8]
params = model.init(rng, input_example, True)

# Optimizer
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Sanity check (eval mode, no dropout randomness)
y_example = model.apply(params, rng, input_example, False)
print("Example output MSE:", np.mean((np.asarray(y_example) - y_train[:8])**2))

# ---------------------------------------------------------------
# Loss / Metrics (loss kept OUTSIDE train_step)
# ---------------------------------------------------------------
def mse_loss(params, rng, x, y, is_training: bool):
    preds = model.apply(params, rng, x, is_training)  # rng used only if training (dropout)
    return jnp.mean(abs(preds-y))#jnp.mean((preds - y) ** 2)

@jax.jit
def train_step(params, opt_state, rng, x, y):
    rng, subkey = random.split(rng)
    loss, grads = jax.value_and_grad(mse_loss, argnums=0)(params, subkey, x, y, True)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, rng, loss

@jax.jit
def eval_loss(params, rng, x, y):
    return mse_loss(params, rng, x, y, False)

def predict(params, x_np):
    """
    Predict in original target units.
    x_np: numpy array (N, features) *UNSCALED*. We'll apply x_scaler.
    """
    x_scaled = x_scaler.transform(x_np.astype(np.float32))
    preds_norm = model.apply(params, random.PRNGKey(0), jnp.array(x_scaled), False)
    return np.asarray(preds_norm)

# ---------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------
patience = 5000          # epochs without improvement before early stop (set None to disable)
min_delta = 5           # improvement threshold in MSE (tune to your scale)
best_val = float("inf")
epochs_no_improve = 0
best_params_snapshot = None

batch_size = 32
num_epochs = 25000
warmup_epochs = 1000

train_losses = []
test_losses = []

N = X_train_jnp.shape[0]

for epoch in range(1, num_epochs + 1):
    # Shuffle each epoch
    perm = np.random.permutation(N)

    # Mini-batch training
    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        x_b = X_train_jnp[idx]
        y_b = y_train_jnp[idx]
        params, opt_state, rng, loss = train_step(params, opt_state, rng, x_b, y_b)

    # Compiled eval (dropout OFF)
    train_l = float(eval_loss(params, rng, X_train_jnp, y_train_jnp).item())
    test_l  = float(eval_loss(params, rng, X_test_jnp,  y_test_jnp ).item())
    train_losses.append(train_l)
    test_losses.append(test_l)

    if epoch % 100 == 0 or epoch == 1:
        print(f"Epoch {epoch:5d} | Train RMSE: {train_l**0.5:.6f} | Test RMSE: {test_l**0.5:.6f}")

    # ---- Checkpoint on validation improvement ----
    if epoch >= warmup_epochs and (test_l + min_delta < best_val):
        best_val = test_l
        epochs_no_improve = 0

        # (optional) keep a RAM copy too
        best_params_snapshot = jax.tree.map(lambda a: a, params)

        # enrich config with metrics/epoch
        extra_cfg = {
            "exp_type": EXP_TYPE,
            "feature_columns": COLUMNS[EXP_TYPE],
            "architecture": Detector_architecture,
            "activation": activation,
            "dropout_rate": DROPOUT_RATE,
            "seed": 42,
            "x_scaler_class": "StandardScaler",
            "epoch": epoch,
            "best_val_mse_normalized": best_val,
        }
        save_bundle(model_path, params, x_scaler, extra_cfg)
        print(f" Saved new BEST @ epoch {epoch}: val RMSE={best_val**0.5:.6f}")
    else:
        epochs_no_improve += 1
        if patience is not None and epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

# ---------------------------------------------------------------
# Plot losses at the end (Train & Test)
# ---------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(train_losses, label="Train MSE (normalized)")
plt.plot(test_losses, label="Test MSE (normalized)")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training & Test Loss (log scale)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% Load the model bundle (example)
# ---------------------------------------------------------------
bundle = load_bundle(model_path)
params_loaded = bundle["params"]
x_scaler_loaded = bundle["x_scaler"]
cfg_loaded = bundle["config"]

params = params_loaded
x_scaler = x_scaler_loaded

# ---------------------------------------------------------------
# Final report (in original units)
# ---------------------------------------------------------------


# Eval-mode predictions (dropout OFF)
preds_norm_test = model.apply(params, random.PRNGKey(0), X_test_jnp, False)
#preds_norm_val  = model.apply(params, random.PRNGKey(1), jnp.array(X_val_arr), False)

preds     = np.asarray(preds_norm_test)
# preds_val = np.asarray(preds_norm_val)
rmse      = np.sqrt(np.mean((preds - y_test_sr.values.reshape(-1, 1)) ** 2))

print(f"\nFinal Test RMSE (original units): {rmse:.6f}")
# print(f"Sample predictions (original units): {preds_val.squeeze()} vs {preds.squeeze()}")


plt.figure()
plt.scatter(y_test_sr, preds)
plt.plot([y_test_sr.min(), y_test_sr.max()], [y_test_sr.min(), y_test_sr.max()], "r--")
plt.xlabel(f"Actual {FOLDER}")
plt.ylabel(f"Predicted {FOLDER}")
plt.title(f"Actual vs Predicted {FOLDER} ({EXP_TYPE})")
plt.text(
    0.05, 0.95,
    f"Test RMSE: {rmse:.3f}",
    transform=plt.gca().transAxes,
    verticalalignment='top'
)
plt.tight_layout()

plt.show()

# %%
# ---------------------------------------------------------------
# Compare two trained models on a grid and plot

fluid_volume = [60,100,250,500, 750, 1000] # ml
#fluid_volume = [500]
# Get average of actual data at this fluid volume
#data_100ml = data[data['fluid_volume'] == fluid_volume].groupby('agitation_rate (cpm)').agg(lambda x: x.mean() if x.dtype.kind in 'biufc' else x.iloc[0]).reset_index()
data_subset = (
    data[data['fluid_volume'].isin(fluid_volume)]
    .groupby(['fluid_volume', 'agitation_rate (cpm)'])
    .agg(lambda x: x.mean() if x.dtype.kind in 'biufc' else x.iloc[0])
    .reset_index()
)
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
    fluid_volume_ml = fluid_volume  # ml
    Dye_volume = fluid_volume_ml[0] * 0.48  # ul
    Injection_time = 27
    flow_rate = Dye_volume * 1e-3 / Injection_time * 60  # ul/s

    # 10..60 cpm (step 1)
    agitation_rate_cpm = range(1, 61, 1)

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

dummy_data = build_base_design()



X_scaled = x_scaler.transform(dummy_data[COLUMNS[EXP_TYPE]].values.astype(np.float32))
preds = model.apply(params, rng, jnp.array(X_scaled), False)
dummy_data["pred"] = np.asarray(preds)
dummy_data['Ntmin'] = dummy_data['pred'] * (dummy_data['agitation_rate (cpm)']/60)
data_subset['Ntmin'] = data_subset['y'] * (data_subset['agitation_rate (cpm)']/60)


mask = (
    ((dummy_data['fluid_volume'] == 1000) &
     (dummy_data['agitation_rate (cpm)'] <= 45))
    |
    ((dummy_data['fluid_volume'] == 750) &
     (dummy_data['agitation_rate (cpm)'] <= 25))
    |
    ((dummy_data['fluid_volume'] == 500) &
    (dummy_data['agitation_rate (cpm)'] <= 20))
    |
    ((dummy_data['fluid_volume'] == 250) &
     (dummy_data['agitation_rate (cpm)'] <= 5)))


dummy_data = dummy_data[~mask].copy()

#%%
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=dummy_data,
    x="agitation_rate (cpm)",
    y="pred",
    hue="fluid_volume",
    palette="viridis",
    s=100,
)
sns.scatterplot(
    data=data_subset,
    x="agitation_rate (cpm)",
    y="y",
    hue = "fluid_volume",
    s=150,
    #label="Actual 100ml Data",
    marker="X",
    palette="bright",
    alpha =0.9
)

#plt.yscale('log')
plt.xlabel("Agitation Rate (cpm)")
plt.ylabel(f"Predicted {FOLDER} [s]")
plt.title(f"Predicted {FOLDER} vs Agitation Rate ({EXP_TYPE})")
plt.legend(title="Fluid Volume (ml)", loc= 'upper left')   
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()
plt.savefig('mixing_time_prediction.png', dpi=500, bbox_inches='tight')

# %%
