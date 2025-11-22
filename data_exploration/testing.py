# import pickle
# import numpy as np
# from huggingface_hub import hf_hub_download

# # Download the static variables from HuggingFace.
# # static_path = hf_hub_download(
# #     repo_id="microsoft/aurora",
# #     filename="aurora-0.25-wave-static.pickle",
# # )
# static_path = hf_hub_download(
#     repo_id="microsoft/aurora",
#     filename="aurora-0.4-air-pollution-static.pickle",
# )
# print("Static variables downloaded!")

# with open(static_path, "rb") as f:
#     static_vars: dict[str, np.ndarray] = pickle.load(f)

# for var, value in static_vars.items():
#     print(f"shape of {var=} {value.shape}")

import xarray as xr

with xr.open_dataset(
    "reanalysis-era5-single-levels\hydrological_full_year\data_stream-oper_stepType-instant.nc",
    engine="netcdf4",
) as era5:
    print(era5)
