import os
import pandas as pd
import xarray as xr
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

def silver(bronze_netcdf, dir):
    # import netcdf file
    netcdf_filename = os.path.join(dir, bronze_netcdf)
    ds = xr.open_dataset(netcdf_filename)
    silver_name = netcdf_filename.replace(".nc", "_silver.nc")

    # Calculate Uship and Vship the ship velocity components in m/s
    Uship = np.gradient(ds.LAT) * 60 * 1.852e4 / ( np.gradient(ds.datetime).astype('float64') * 1e-9)
    Vship = np.gradient(ds.LON) * 60 * np.cos(np.deg2rad(ds.LAT)) * 1.852e4 / (np.gradient(ds.datetime).astype('float64') * 1e-9)

    # calculate alpha and beta, the compass correction parameters
    alpha = np.arctan2(np.mean(ds.BTNorth * Vship - ds.BTEast * Uship), np.mean(ds.BTNorth * Uship + ds.BTEast * Vship))
    beta = np.sqrt(np.mean(Uship**2 + Vship**2) / (np.mean(ds.BTNorth**2 + ds.BTEast**2))) - 1
    print(f"Alpha: {np.rad2deg(alpha).values} degrees, Beta: {beta.values}")

    Ucorr = (1 + beta * (ds.u_bm * np.cos(alpha) - ds.v_bm * np.sin(alpha)))
    Vcorr = (1 + beta * (ds.u_bm * np.sin(alpha) + ds.v_bm * np.cos(alpha)))

    # build silver dataset, to do so drop BTNorth and BTEast and replace u_bm and v_bm with u_corr and v_corr
    # as well as rename

    ds_silver = ds.drop_vars(["BTEast", "BTNorth",'u_bm','v_bm']).assign_coords(u_corr=Ucorr, v_corr=Vcorr)


    # save to new netcdf
    print(f"Saving silver data to {silver_name}")
    ds_silver.to_netcdf(silver_name)
    print("Silver data saved successfully.")

    return silver_name, netcdf_filename, dir

# silver("data/adcp_data/108191_20260611T161412UTC_bronze.nc")

def silver_plots(silver, bronze, directory):
    silver = xr.open_dataset(silver)
    bronze = xr.open_dataset(bronze)

    fig_dir = os.path.join(directory, "silver")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if not os.path.exists(fig_dir):
        print(f"Error: Could not create directory {fig_dir}")
        sys.exit(1)

    # silver total track
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(silver.LON, silver.LAT, 'o', markersize=1, color='red')
    ax.set_title('Total Track')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.savefig(os.path.join(fig_dir, "total_track.png"))

    # silver time vs latitude
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(silver.datetime, silver.LAT, 'o', markersize=1, color='blue')
    ax.set_title('Time vs Latitude')
    ax.set_xlabel('Time')
    ax.set_ylabel('Latitude')
    plt.savefig(os.path.join(fig_dir, "time_vs_latitude.png"))

    # silver time vs longitude
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(silver.datetime, silver.LON, 'o', markersize=1, color='green')
    ax.set_title('Time vs Longitude')
    ax.set_xlabel('Time')
    ax.set_ylabel('Longitude')
    plt.savefig(os.path.join(fig_dir, "time_vs_longitude.png"))

    # Velocities - two plots with u velocities on the top plot and v velocites on the bottom plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    ax[0].plot(bronze.u_bm, silver.u_corr, 'o', markersize=1, color='black')
    ax[0].plot(bronze.v_bm, silver.v_corr, 'o', markersize=1, color='blue')
    # add line representing y=x
    ax[0].plot([bronze.u_bm.min(), bronze.u_bm.max()], [bronze.u_bm.min(), bronze.u_bm.max()], 'r--', label='y=x')
    ax[0].set_title('U and V Velocities')
    ax[0].set_xlabel('Bronze (m/s)')
    ax[0].set_ylabel('Silver (m/s)')
    ax[0].legend(['U', 'V'])

    ax[1].plot(bronze.datetime, bronze.u_bm.mean(dim='bin_depth'), 'o', markersize=1, color='black')
    ax[1].plot(bronze.datetime, bronze.v_bm.mean(dim='bin_depth'), 'o', markersize=1, color='blue')
    ax[1].plot(silver.datetime, silver.u_corr.mean(dim='bin_depth'), 'o', markersize=1, color='red')
    ax[1].plot(silver.datetime, silver.v_corr.mean(dim='bin_depth'), 'o', markersize=1, color='green')
    ax[1].set_title('U and V Velocities over Time')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Velocity (m/s)')
    ax[1].legend(['Bronze U', 'Bronze V', 'Silver U', 'Silver V'])
    plt.savefig(os.path.join(fig_dir, "velocities.png"))

    print("Silver plots saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert bronze netCDF file to silver netCDF file and generate plots.")
    parser.add_argument("file_path", type=str, help="Directory path of the bronze netCDF file")
    parser.add_argument("file_name", type=str, help="Name of the bronze netCDF file")
    args = parser.parse_args()

    plot_const = silver(args.file_name, args.file_path)
    silver_plots(plot_const[0], plot_const[1], plot_const[2])