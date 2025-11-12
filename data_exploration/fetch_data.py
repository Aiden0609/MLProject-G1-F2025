import cdsapi

client = cdsapi.Client()

dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
    ],
    "year": ["2023"],
    "month": ["01"],
    "day": ["01"],
    "time": ["00:00", "06:00", "12:00", "18:00"],
    "pressure_level": [
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
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [90, -180, -90, 180],
}

target = f"{dataset}.nc"  # Output file. Adapt as you wish.

client.retrieve(dataset, request, target)
