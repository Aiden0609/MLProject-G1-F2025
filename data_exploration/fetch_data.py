import cdsapi

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
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "2m_temperature",
        "mean_sea_level_pressure",
        "sea_surface_temperature",
        "surface_pressure",
        "air_density_over_the_oceans",
        "downward_uv_radiation_at_the_surface",
        "surface_latent_heat_flux",
        "instantaneous_surface_sensible_heat_flux",
        "surface_sensible_heat_flux",
        "surface_net_thermal_radiation",
    ],
    "year": ["2023"],
    "month": ["01"],
    "day": ["01", "02", "03"],
    "time": ["00:00", "06:00", "12:00", "18:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    # "area": [66, -60, 48, -24],
    "area": [90, -180, -90, 180],
}

target = f"{dataset}.nc"  # Output file. Adapt as you wish.

client.retrieve(dataset, request).download()
