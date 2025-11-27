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
dataset_single_levels = "reanalysis-era5-single-levels"
dataset_pressure_levels = "reanalysis-era5-pressure-levels"
datapath = "data"
if not os.path.exists(datapath):
    os.mkdir(datapath)

instant_variables = [
    "sea_surface_temperature",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    # "instantaneous_surface_sensible_heat_flux",
    # "instantaneous_10m_wind_gust",
    # "soil_temperature_level_1",
    # "temperature_of_snow_layer",
    # "2m_dewpoint_temperature",
    # "mean_sea_level_pressure",
    # "surface_pressure",
    # "ice_temperature_layer_1",
    "sea_ice_cover",
    "mean_sea_level_pressure",
    # "skin_temperature",
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
hydrological_variables = [
    "potential_evaporation",
    "runoff",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
]

flux_variables = [
    "mean_surface_direct_short_wave_radiation_flux",
    "mean_surface_direct_short_wave_radiation_flux_clear_sky",
    "mean_surface_downward_long_wave_radiation_flux",
    "mean_surface_downward_long_wave_radiation_flux_clear_sky",
    "mean_surface_downward_short_wave_radiation_flux",
    "mean_surface_downward_short_wave_radiation_flux_clear_sky",
    "mean_surface_downward_uv_radiation_flux",
    "mean_surface_latent_heat_flux",
    "mean_surface_net_long_wave_radiation_flux",
    "mean_surface_net_long_wave_radiation_flux_clear_sky",
    "mean_surface_net_short_wave_radiation_flux",
    "mean_surface_net_short_wave_radiation_flux_clear_sky",
    "mean_surface_sensible_heat_flux",
]
static_variables = [
    "geopotential",
    "land_sea_mask",
    "soil_type",
]
atmos_variables = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
]
pressure_levels = [
    "50",
    "100",
    "150",
    "200",
    "250",
    "300",
    "400",
    "500",
    "600",
    "700",
    "850",
    "925",
    "1000",
]
variables = {
    "surface-level": instant_variables,
    "static": static_variables,
    "atmospheric": atmos_variables,
    # "wave_instant": wave_instant_variables,
    # "accumulated": accumulated_variables,
    # "hydrological": hydrological_variables,
    # "flux": flux_variables,
    # "invariant": invariant_variables,
}


for name, variable in variables.items():
    # name = name + "_full_year"
    request = {
        "product_type": ["reanalysis"],
        "variable": variable,
        "year": ["2023"],
        # "month": ["01", "04", "07", "11"],
        "month": ["01"],
        # "day": ["01", "02", "03"],
        "day": ["01", "08", "15", "24"],
        # "time": ["00:00", "06:00", "12:00", "18:00"],
        "time": ["06:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        # "area": [66, -60, 48, -24],
        # "area": [90, -180, -90, 180],
    }
    if name.startswith("atmospheric"):
        request["pressure_level"] = pressure_levels
        dataset = dataset_pressure_levels
    else:
        dataset = dataset_single_levels
        if name.startswith("static"):
            request["day"] = ["01"]
            request["time"] = ["06:00"]

    target = f"{datapath}/{name}.nc"  # Output file. Adapt as you wish.

    client.retrieve(dataset, request).download(target)
    # Source - https://stackoverflow.com/a
    # Posted by ClimateUnboxed, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-11-26, License - CC BY-SA 4.0
    # os.system(f"cdo sellonlatbox,0,360,-90,90 {target} {datapath}/{name}_remapped.nc")
    if name.startswith("wave_instant"):

        # cwd = os.getcwd()
        os.system(
            f"cdo remapbil,{datapath}/instant.nc {target} {datapath}/bil_remapped_{name}.nc"
        )
