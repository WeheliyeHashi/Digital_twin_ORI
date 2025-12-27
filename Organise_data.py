# %%

import pandas as pd
from pathlib import Path
import os
import numpy as np

# %%


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
        "Mixing time [s]",
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
        # "MCV_of_single_cell (ml)",
        # "number_cells",
        # "cell_vol/fluid_vol",
        "fluid_viscosity (mPas)",
        "fluid_density (kg/m^3)",
        "Mixing time [s]",
    ],
}

exp_type = "Rock"
Folder_type = "Mixing_time"
columns = COLUMNS[exp_type]
mixing_time_path = Path(r"Mixing_time\Rock\Mixing_time_G_updated.xlsx").resolve()

df = pd.read_excel(mixing_time_path)[columns]

mask = (
    (df["up_dwell"] == 0)
    & (df["down_dwell"] == 0)
    & (df["bio_diameter (mm)"] == 140)
    & (np.isclose(df["split_period"], 0.5))
    & (df["start_angle (deg)"] == -10)
    & (df["stop_angle (deg)"] == 10)
    & (np.isclose(df["fluid_viscosity (mPas)"], 1e-3))
    & (df["fluid_density (kg/m^3)"] == 1000)
    & (df["base_height (mm)"] == 0)
)

df_fixed = df[mask].copy()

# %% check if the following conditions and save to save_path folder


save_path = Path(os.path.join(Folder_type, "model_2", "mixing_data.xlsx"))

# Make sure the directory exists
save_path.parent.mkdir(parents=True, exist_ok=True)

# Save the Excel file
df_fixed.to_excel(save_path, index=False)


# %%
