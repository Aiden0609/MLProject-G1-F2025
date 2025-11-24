import torch
import numpy as np  


d_srf_abr2full = {"2t": "2m_temperature", 
                  "10u": "10m_u_component_of_wind", 
                  "10v": "10m_v_component_of_wind", 
                  "msl": "mean_sea_level_pressure", 
                  "tp": "total_precipitation",
                  "tp_mswep": "total_precipitation_MSWEP",
                  "pe": "potential_evaporation",
                  "e": "evaporation",
                  "r": "runoff",
                  "swvl": "volumetric_soil_water_layer",
                  "swc": "soil_water_content",
                  "sst": "sea_surface_temperature",
                  "slhf": "surface_latent_heat_flux",
                  "sshf": "surface_sensible_heat_flux",
                  "ssr": "surface_net_solar_radiation", 
                  "str": "surface_net_thermal_radiation",
                  "ssrd": "surface_solar_radiation_downwards",
                  "strd": "surface_thermal_radiation_downwards",
                  "tsr": "top_net_solar_radiation",
                  "ttr": "top_net_thermal_radiation",
                  "tisr": "toa_incident_solar_radiation",
                  "sst": "sea_surface_temperature",
                  }


def transform_data(data, var_name, eps=1e-5, direct=True): # transformation follows [Rasp 2020], also used in FourCastNet
    if np.isin(var_name, ["tp", "tp_mswep", "r"]):
        if direct:
            if isinstance(data, torch.Tensor):
                return torch.log(1 + data/eps)
            else:
                return np.log(1 + data/eps)
        else:
            if isinstance(data, torch.Tensor):
                return eps*(torch.exp(data) - 1)
            else:
                return eps*(np.exp(data) - 1)
            
    elif np.isin(var_name, ["pe", "e"]):
        if direct:
            return -5e3*data
        else:
            return data/(-5e3)

    elif np.isin(var_name, ["ssr", "ssrd", "strd", "tsr", "ttr", "str", "tisr", "slhf", "sshf"]):
        if direct:
            return 1e-6*data
        else:
            return 1e6*data
        
    else:
        return data