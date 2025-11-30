from pathlib import Path
import os
import cdsapi
import time
import random
import xarray as xr

# download path
download_path = Path(f"/scratch/{os.environ['USER']}/data/finetune-data-2020-2024")
download_path = download_path.expanduser()
download_path.mkdir(parents=True, exist_ok=True)

c = cdsapi.Client()

years = ["2020", "2021", "2022", "2023", "2024"]
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
days = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
]
times = ["00:00", "06:00", "12:00", "18:00"]


# retry logic
def robust_retrieve(dataset, request_dict, outfile, max_retries=8):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    attempt {attempt}/{max_retries}")
            c.retrieve(dataset, request_dict, str(outfile))
            print("    success")
            return True
        except Exception as e:
            print(f"    error: {e}")
            if attempt == max_retries:
                print(f"    failed permanently: {outfile}")
                return False
            sleep_time = min(60 * attempt, 600) + random.uniform(0, 5)
            print(f"    waiting {sleep_time:.1f}s before retry")
            time.sleep(sleep_time)
    return False


# check if a file is readable
def print_dimensions(path) -> bool:
    try:
        ds = xr.open_dataset(path)
        print(f"    dims: {dict(ds.dims)}")
        ds.close()
        return True
    except OSError as e:
        print(f"    CORRUPT: file must be redownloaded")
        os.remove(path)
        return False
    except Exception as e:
        print(f"    CORRUPT: cannot open ({e})")
        return True


# static data
static_file = download_path / "static.nc"
if not static_file.exists():
    print("downloading static variables")
    robust_retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["geopotential", "land_sea_mask", "soil_type"],
            "year": "2020",
            "month": "01",
            "day": "01",
            "time": "00:00",
            "format": "netcdf",
        },
        static_file,
    )
    print_dimensions(static_file)
else:
    print("static file exists")

# surface variables
surface_vars = [
    "sea_surface_temperature",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "sea_ice_cover",
    "mean_sea_level_pressure",
]

for year in years:
    print(f"\n=== surface {year} ===")

    for m in months:
        out_file = download_path / f"era5_surface_{year}_{m}.nc"

        if out_file.exists():
            print(f"  exists: {out_file.name}")
            not_corrupt = print_dimensions(out_file)
            if not_corrupt:
                continue

        print(f"  downloading {out_file.name}")
        start = time.time()
        ok = robust_retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": surface_vars,
                "year": year,
                "month": m,
                "day": days,
                "time": times,
                "format": "netcdf",
            },
            out_file,
        )
        end = time.time()
        if (end - start) // 60 > 10:
            c = cdsapi.Client()

        print_dimensions(out_file)


# atmospheric variables
atmospheric_vars = [
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

for year in years:
    print(f"\n=== pressure {year} ===")

    for m in months:
        out_file = download_path / f"era5_atmospheric_{year}_{m}.nc"

        if out_file.exists():
            print(f"  exists: {out_file.name}")
            print_dimensions(out_file)
            continue

        print(f"  downloading {out_file.name}")
        start = time.time()
        ok = robust_retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": atmospheric_vars,
                "pressure_level": pressure_levels,
                "year": year,
                "month": m,
                "day": days,
                "time": times,
                "format": "netcdf",
            },
            out_file,
        )
        end = time.time()
        if (end - start) // 60 > 10:
            c = cdsapi.Client()

        print_dimensions(out_file)
