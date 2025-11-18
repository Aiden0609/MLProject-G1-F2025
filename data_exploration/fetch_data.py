import cdsapi
import os

client = cdsapi.Client()

# dataset = "reanalysis-era5-pressure-levels"
# request = {
#     "product_type": ["reanalysis"],
#     "variable": [
#         "geopotential",
#         "specific_humidity",
#         "temperature",
#         "u_component_of_wind",
#         "v_component_of_wind",
#     ],
#     "year": ["2023"],
#     "month": ["01"],
#     "day": ["01"],
#     "time": ["00:00", "06:00", "12:00", "18:00"],
#     "pressure_level": [
#         "50",
#         "100",
#         "150",
#         "200",
#         "250",
#         "300",
#         "400",
#         "500",
#         "600",
#         "700",
#         "850",
#         "925",
#         "1000",
#     ],
#     "data_format": "netcdf",
#     "download_format": "unarchived",
#     "area": [90, -180, -90, 180],
# }
dataset = "reanalysis-era5-single-levels"
if not os.path.exists(dataset):
    os.mkdir(dataset)

instant_variables = [
    "sea_surface_temperature",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "instantaneous_surface_sensible_heat_flux",
    "2m_dewpoint_temperature",
    "mean_sea_level_pressure",
    "surface_pressure",
    "skin_temperature",
]

wave_instant_variables = [
    "air_density_over_the_oceans",
]

accumulated_variables = [
    "downward_uv_radiation_at_the_surface",
    "surface_latent_heat_flux",
    "surface_sensible_heat_flux",
    "surface_net_thermal_radiation",
]
variables = {
    "instant": instant_variables,
    # "wave_instant": wave_instant_variables,
    # "accumulated": accumulated_variables,
}
for name, variable in variables.items():
    request = {
        "product_type": ["reanalysis"],
        "variable": variable,
        "year": ["2023"],
        "month": ["01"],
        "day": ["01", "02", "03"],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        # "area": [66, -60, 48, -24],
        "area": [90, -180, -90, 180],
    }

    target = f"{dataset}/{name}.nc"  # Output file. Adapt as you wish.

    client.retrieve(dataset, request).download(target)
