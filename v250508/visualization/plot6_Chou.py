import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import datetime

#
output_folder = '/work/choutilin1/out_vars22a/'
plot_file_name = 'out_0000.nc'  #
plot_file_path1 = os.path.join(output_folder, plot_file_name)
# plot_file_path2 = os.path.join(output_folder,'sfno-rain.nc')

#
coast = pd.read_csv("/home/choutilin1/visualization/coast.csv")

#
save_folder = '250627'
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)

#
precip_lev =  [0, 0.5, 1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130,150,200,300,400]
precip_color = [
    "#fdfdfd",  # 0.01 - 0.10 inches
    "#c9c9c9",  # 0.10 - 0.25 inches
    "#9dfeff",
    "#01d2fd",  # 0.25 - 0.50 inches
    "#00a5fe",  # 0.50 - 0.75 inches
    "#0177fd",  # 0.75 - 1.00 inches
    "#27a31b",  # 1.00 - 1.50 inches
    "#00fa2f",  # 1.50 - 2.00 inches
    "#fffe33",  # 2.00 - 2.50 inches
    "#ffd328",  # 2.50 - 3.00 inches
    "#ffa71f",  # 3.00 - 4.00 inches
    "#ff2b06",
    "#da2304",  # 4.00 - 5.00 inches
    "#aa1801",  # 5.00 - 6.00 inches
    "#ab1fa2",  # 6.00 - 8.00 inches
    "#db2dd2",  # 8.00 - 10.00 inches
    "#ff38fb",  # 10.00+
    "#ffd5fd"]

#
lat = np.linspace(-90, 90, 721)
lon = np.linspace(0, 359.75, 1440)

# 限定的經緯度範圍
lat_min = np.argwhere(lat == -25)[0][0]
lat_max = np.argwhere(lat == 30)[0][0]
lon_min = np.argwhere(lon == 75)[0][0]
lon_max = np.argwhere(lon == 170)[0][0]

lat = lat[lat_min:lat_max]
lon = lon[lon_min:lon_max]
lon_map = coast['lon_map'].to_numpy().flatten()
lat_map = coast['lat_map'].to_numpy().flatten()

#
example_batch = xr.load_dataset(plot_file_path1).compute()
# rain_batch= xr.load_dataset(plot_file_path2).compute()
#

vname = {
'time':'dim0',
'lead_time':'dim0',
't2m':'var3',
'u10m':'var0',
'v10m':'var1',
'msl':'var2',
'u850':'var9',
'v850':'var10',
'u500':'var14',
'v500':'var15',
'z850':'var11',
'z500':'var16',
't850':'var8',
}


time_in_hours = example_batch[vname['time']].values[0]  
# start_time = example_batch[vname['time']].values[0]
start_time = np.datetime64('2015-01-01T00:00')

#
for lead_time in range(example_batch.dims[vname['lead_time']]):
    #
    lead_time_hours = 6 * lead_time  #
    target_time = start_time + np.timedelta64(lead_time_hours, 'h')  #
    time_str = np.datetime_as_string(target_time, unit='h')  #
    #
    fig, axs = plt.subplots(3, 2, figsize=(22, 15), dpi=400)

    #

    # precip  = np.flip(rain_batch.variables[vname['tp06']][0, lead_time, :, :])[lat_min:lat_max, lon_min:lon_max]
    
   
    t2m = np.flip(example_batch.variables[vname['t2m']][lead_time, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max] - 273.15  #
    u10 = np.flip(example_batch.variables[vname['u10m']][lead_time, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
    v10 = np.flip(example_batch.variables[vname['v10m']][lead_time, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
    
    
    mslp = np.flip(example_batch.isel(dim0=lead_time).variables[vname['msl']].values[lat_min:lat_max, lon_min:lon_max], axis=0) / 100
    u850 = np.flip(example_batch.isel(dim0=lead_time).variables[vname['u850']].values[lat_min:lat_max, lon_min:lon_max], axis=0)
    v850 = np.flip(example_batch.isel(dim0=lead_time).variables[vname['v850']].values[lat_min:lat_max, lon_min:lon_max], axis=0)
    u500 = np.flip(example_batch.isel(dim0=lead_time).variables[vname['u500']].values[lat_min:lat_max, lon_min:lon_max], axis=0)
    v500 = np.flip(example_batch.isel(dim0=lead_time).variables[vname['v500']].values[lat_min:lat_max, lon_min:lon_max], axis=0)
    # u200 = np.flip(example_batch.isel(dim0=lead_time).variables[vname['u250']].values[lat_min:lat_max, lon_min:lon_max], axis=0)
    # v200 = np.flip(example_batch.isel(dim0=lead_time).variables[vname['v250']].values[lat_min:lat_max, lon_min:lon_max], axis=0)
    geopotential_850 = np.flip(example_batch.isel(dim0=lead_time).variables[vname['z850']].values[lat_min:lat_max, lon_min:lon_max] , axis=0)/ 9.81  # divide by 9.81 to convert to gpm
    geopotential_500 = np.flip(example_batch.isel(dim0=lead_time).variables[vname['z500']].values[lat_min:lat_max, lon_min:lon_max], axis=0) / 9.81  # divide by 9.81 to convert to gpm
    # geopotential_200 = np.flip(example_batch.isel(dim0=lead_time).variables[vname['z250']].values[lat_min:lat_max, lon_min:lon_max], axis=0) / 9.81  # divide by 9.81 to convert to gpm
    t850 = np.flip(example_batch.variables[vname['t850']][lead_time, lat_min:lat_max, lon_min:lon_max], axis=0) - 273.15  #
    #
    
    mslp_lev = np.linspace(990, 1035, 30)  #
    geopotential_lev850 = np.arange(1200, 1541, 20)
    geopotential_lev200 = np.arange(10500, 12650, 100)
    geopotential_lev500 = np.arange(5400, 5900, 30)
    wind_speed_lev = np.linspace(0,45 , 35)
    wind_speed_850 = np.sqrt(u850**2 + v850**2)
    wind_speed_500 = np.sqrt(u500**2 + v500**2)
    # wind_speed_200 = np.sqrt(u200**2 + v200**2)
    # wind_speed_200_filtered = np.where(wind_speed_200 > 10, wind_speed_200, np.nan)
    
    wind_speed_500_filtered = np.where(wind_speed_500 > 10, wind_speed_500, np.nan)
    wind_speed_850_filtered = np.where(wind_speed_850 > 10, wind_speed_850, np.nan)
    
    #
    #
    #200
    # cs_wind = axs[0,0].contourf(lon, lat, wind_speed_200_filtered, levels=wind_speed_lev, cmap='YlGn', extend='both')
    # cs = axs[0,0].contour(lon, lat, geopotential_200, levels=geopotential_lev200, colors='black', linewidths=1)
    # quiver = axs[0,0].quiver(lon[::10], lat[::10], u200[::10, ::10], v200[::10, ::10], scale=600, width=0.0025)
    # axs[0,0].clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
    # axs[0,0].plot(lon_map, lat_map, color='k', linewidth=0.7)

    # axs[0,0].set_xlim([75, 170])
    # axs[0,0].set_ylim([-25, 30])

    # fig.colorbar(cs_wind, ax=axs[0,0], orientation='vertical', label='200 hPa Wind Speed (m/s)')
    # axs[0,0].set_title(f'200 hPa geopotential height & 200 wind vector & 200 wind speed\nTime: {time_str}', fontsize=20)

    #500
    cs_wind = axs[0,1].contourf(lon, lat, wind_speed_500_filtered, levels=wind_speed_lev, cmap='YlGn', extend='both')
    cs = axs[0,1].contour(lon, lat, geopotential_500, levels=geopotential_lev500, colors='black', linewidths=1)
    quiver = axs[0,1].quiver(lon[::10], lat[::10], u500[::10, ::10], v500[::10, ::10], scale=600, width=0.0025)
    axs[0,1].clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
    axs[0,1].plot(lon_map, lat_map, color='k', linewidth=0.7)

    axs[0,1].set_xlim([75, 170])
    axs[0,1].set_ylim([-25, 30])

    fig.colorbar(cs_wind, ax=axs[0,1], orientation='vertical', label='500 hPa Wind Speed (m/s)')
    axs[0,1].set_title(f'500 hPa geopotential height & 500 wind vector \n & 500 wind speedTime: {time_str}', fontsize=20)


    #850
    cs_wind = axs[1,0].contourf(lon, lat, wind_speed_850_filtered, levels=wind_speed_lev, cmap='YlGn', extend='both')
    cs = axs[1,0].contour(lon, lat, geopotential_850, levels=geopotential_lev850, colors='black', linewidths=1)
    quiver = axs[1,0].quiver(lon[::10], lat[::10], u850[::10, ::10], v850[::10, ::10], scale=600, width=0.0025)
    axs[1,0].clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
    axs[1,0].plot(lon_map, lat_map, color='k', linewidth=0.7)

    axs[1,0].set_xlim([75, 170])
    axs[1,0].set_ylim([-25, 30])

    fig.colorbar(cs_wind, ax=axs[1,0], orientation='vertical', label='850 hPa Wind Speed (m/s)')
    axs[1,0].set_title(f'850 hPa geopotential height & 850 wind vector & 850 wind speed\nTime: {time_str}', fontsize=20)


    #500gp+850t
    cs = axs[1,1].contour(lon, lat, geopotential_500, levels=geopotential_lev500, colors='black', linewidths=1)
    temp_contour = axs[1,1].contourf(lon, lat, t850, cmap='coolwarm', levels=np.linspace(0,27, 10))
    axs[1,1].clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
    axs[1,1].plot(lon_map, lat_map, color='k', linewidth=0.7)

    axs[1,1].set_xlim([75, 170])
    axs[1,1].set_ylim([-25, 30])

    fig.colorbar(temp_contour, ax=axs[1,1], orientation='vertical', label='Temperature (°C)')
    axs[1,1].set_title(f'500 hPa geopotential height & 850hpa Temperature (°C)\nTime: {time_str}', fontsize=20)


    #mslp+precip
    cs_mslp = axs[2,0].contour(lon, lat, mslp, levels=mslp_lev, colors='black', linewidths=1)
    # cs = axs[2,0].contourf(lon, lat, precip * 1000, levels=precip_lev, colors=precip_color)
    
    axs[2,0].clabel(cs_mslp, inline=True, fontsize=8, fmt='%1.0f')
    axs[2,0].plot(lon_map, lat_map, color='k', linewidth=0.7)
    axs[2,0].set_xlim([75, 170])
    axs[2,0].set_ylim([-25, 30])
    axs[2,0].set_title(f'Total Precipitation (6hr) & mslp\nTime: {time_str}', fontsize=20)
    fig.colorbar(cs, ax=axs[2,0], label='mm')

    #mslp+10m wind+t2m
    cs = axs[2,1].contourf(lon, lat, t2m, cmap='coolwarm', levels=np.linspace(4,35, 10))
    cs_mslp = axs[2,1].contour(lon, lat, mslp, levels=mslp_lev, colors='black', linewidths=1)
    axs[2,1].clabel(cs_mslp, inline=True, fontsize=8, fmt='%1.0f')
    quiver = axs[2,1].quiver(lon[::10], lat[::10], u10[::10, ::10], v10[::10, ::10], scale=400, width=0.0025)


    axs[2,1].plot(lon_map, lat_map, color='k', linewidth=0.7)
    axs[2,1].set_xlim([75, 170])
    axs[2,1].set_ylim([-25, 30])
    fig.colorbar(cs, ax=axs[2,1], orientation='vertical', label='Temperature (°C)')
    
    axs[2,1].set_title(f'Mean sea level pressure & 10m wind & 2m temperature\nTime: {time_str}', fontsize=20)

    


    #
    plt.suptitle(f'fcn Output - Lead Time: {time_str}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'img_{lead_time+1}.png'), bbox_inches='tight')
    
    plt.suptitle(f'fcn Output - Lead Time: {time_str}', fontsize=20)  #
    plt.subplots_adjust(top=0.95)
    plt.close(fig)

    

print("done")

