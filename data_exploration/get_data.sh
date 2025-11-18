# python fetch_data.py

export D_LOC="reanalysis-era5-single-levels"

cdo remapbil,"$D_LOC/instant.nc" "$D_LOC/wave_instant.nc" "$D_LOC/bil_remapped_wave_instant.nc"
# Slower, but maybe more accurate
cdo remapcon,"$D_LOC/instant.nc" "$D_LOC/wave_instant.nc" "$D_LOC/con_remapped_wave_instant.nc"