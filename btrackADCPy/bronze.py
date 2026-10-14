import os

import pandas as pd
import xarray as xr
import sys
import argparse
import numpy as np
import pymatreader as pmr
import matplotlib.pyplot as plt

def convert_csv_to_netcdf(file_path, file_name):
    """
    Convert a CSV file to a NetCDF file using pandas and xarray.

    Parameters:
    - file_path: str, the directory path where the CSV file is located
    - file_name: str, the name of the CSV file

    Returns:
    - netcdf_file: str, the path to the created NetCDF file
    """
    # Construct the full path to the CSV file
    file_full_path = os.path.join(file_path, file_name)

    # Read the .mat file
    mat = pmr.read_mat(file_full_path)['A']

    # Get datetime information from A.Sup
    year = [int(year) for year in mat['Sup']['year']]
    month = [int(month) for month in mat['Sup']['month']]
    day = [int(day) for day in mat['Sup']['day']]
    hour = [int(hour) for hour in mat['Sup']['hour']]
    minute = [int(min) for min in mat['Sup']['minute']]
    second = [int(sec) for sec in mat['Sup']['second']]
    sec100 = [int(sec100) for sec100 in mat['Sup']['sec100']] # one-hundredths of a second
    # combine the above into a single date-time string and then convert to datetime
    datetime_string = [f"{year}-{month}-{day}T{hour}:{minute}:{second}.{sec100}" for year, month, day, hour, minute, second, sec100 in zip(year, month, day, hour, minute, second, sec100)]
    datetime = pd.to_datetime(datetime_string, format='%Y-%m-%dT%H:%M:%S.%f') # datetime will be used as the index for the dataframe that will be built

    # Get bottom track velocities
    BTEast = mat['Nav']['bvEast_ms']
    BTNorth = mat['Nav']['bvNorth_ms']

    # Get lat and lon
    LAT = mat['Nav']['lat_deg']
    LON = mat['Nav']['long_deg']

    # get bin depth and corresponding water velocities
    bin_depth = mat['Wat']['binDepth']
    u_bm = mat['Wat']['vEast_ms']
    v_bm = mat['Wat']['vNorth_ms']
    percentgood = mat['Wat']['percentGood']
    error_velocity = mat['Wat']['vError_ms']

    depth = mat['Nav']['dsDepth']

    backscatter = mat['Wat']['backscatter']

    temp = mat['Sensor']['temp_degC']



    # combine bin depth and u_bm, v_bm and percentgood into a single dataframe, with datetime as index using xarray
    ds = xr.Dataset({
        'datetime': datetime,
        'BTEast': BTEast,
        'BTNorth': BTNorth,
        'LAT': LAT,
        'LON': LON,
        'bin_depth': (('bin_depth','datetime'),bin_depth),
        'u_bm': (('bin_depth','datetime'),u_bm),
        'v_bm': (('bin_depth','datetime'),v_bm),
        'percentgood': (('bin_depth','datetime'),percentgood),
        'depth': depth,
        'backscatter': (('bin_depth','datetime','beam'),backscatter),
        'temp': temp,
        'error_velocity': (('bin_depth','datetime'),error_velocity),
    })

    # # set index as datetime
    # ds = ds.set_index(datetime='datetime')


    # Define the output NetCDF file path
    netcdf_file = file_full_path.replace(".mat", ".nc")

    # 1. QC: Bottom-track speed (cm/s) should be greater than 10 cm/s
    bts_threshold = 10.0  # cm/s
    bts = np.sqrt(ds.BTEast**2 + ds.BTNorth**2) * 100  # Convert to cm/s

    bts_mask = bts <= bts_threshold

    if bts_mask.any():
        print(f"Warning: {bts_mask.sum().item()} rows have Bottom-track speed <= {bts_threshold} cm/s. These rows will be masked.")
        # Optionally, you can drop these rows from the dataset if you don't want them in the final NetCDF file, allow the user to choose whether to drop or not
        print("Do you want to drop these rows from the dataset? (y/n): ", end="")
        user_input = input().strip().lower()
        if user_input == 'y':
            ds = ds.where(~bts_mask, drop=True)
            print(f"Dropped {bts_mask.sum().item()} rows from the dataset.")
            # # For consistency, we can still add the mask as a new variable to the dataset, even though the rows have been dropped
            # ds = ds.assign(BottomTrackSpeedMask=bts_mask)
        else:
            # print("Rows will be flagged but not dropped.")
            print(f'Nothing will be done with the rows with Bottom-track speed <= {bts_threshold} cm/s.')
            # Add the mask as a new variable to the dataset
            # ds = ds.assign(BottomTrackSpeedMask=bts_mask)
    else:
        print("All rows have Bottom-track speed greater than the threshold.")
        # For consistency, the mask is still added to the dataset.
        # ds = ds.assign(BottomTrackSpeedMask=bts_mask)

    # 2. APPLY STREAMLINING CRITERIA:
    # Streamlining criteria from Valle-Levinson (2024):
    # Error velocity should be less than 10
    # Percent good should be greater than 70
    # Discharge should be less than 100 (not used for now)
    #TODO the validity mask has an issue where its not correctly dropping all items only datetimes and depths, this is
    # likely because of an issue with how the data system is sctructred and linked and needs to be addressed.
    # valid_mask = (ds.error_velocity <= 10) & (ds.percentgood >= 70)

    # ds = ds.where(valid_mask, drop=True)


    # add a new variable to the dataset to indicate which rows have a Bottom-track speed less than or equal to the threshold (True for those rows, False otherwise)
    ds = ds.assign(BottomTrackSpeedMask=bts_mask)

    # Save the xarray Dataset to a NetCDF file
    ds.to_netcdf(netcdf_file)

    return netcdf_file


def create_diagnostic_plots(netcdf_file):
    # generate and save figures into directory named after the netcdf file name and in a sub-driectory called bronze, create if they do not exist

    main_dir = os.path.dirname(netcdf_file.replace(".nc", ""))
    fig_dir = os.path.join(main_dir, "bronze")
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    if not os.path.exists(main_dir):
        print(f"Error: Could not create directory {main_dir}")
        sys.exit(1)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if not os.path.exists(fig_dir):
        print(f"Error: Could not create directory {fig_dir}")
        sys.exit(1)


    ds = xr.open_dataset(netcdf_file)


    # make plots of: total track (from lat, lon), time vs latitude, time vs bottom trakc north velocity, time vs bottom track east velocity
    # north velocity vs bin depth, time vs depth, time vs backscatter, time vs temperature

    # plot lat lon
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ds.LON, ds.LAT, 'o', markersize=1, color='red')
    ax.set_title('Total Track')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.savefig(os.path.join(fig_dir, "total_track.png"))

    # plot time vs lat
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ds.datetime, ds.LAT, 'o', markersize=1, color='blue')
    ax.set_title('Time vs Latitude')
    ax.set_xlabel('Time')
    ax.set_ylabel('Latitude')
    plt.savefig(os.path.join(fig_dir, "time_vs_latitude.png"))

    # plot time vs bottom track north velocity
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ds.datetime, ds.BTNorth, 'o', markersize=1, color='green')
    ax.set_title('Time vs Bottom Track North Velocity')
    ax.set_xlabel('Time')
    ax.set_ylabel('Bottom Track North Velocity')
    plt.savefig(os.path.join(fig_dir, "time_vs_bottom_track_north_velocity.png"))

    # plot time vs bottom track east velocity
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ds.datetime, ds.BTEast, 'o', markersize=1, color='orange')
    ax.set_title('Time vs Bottom Track East Velocity')
    ax.set_xlabel('Time')
    ax.set_ylabel('Bottom Track East Velocity')
    plt.savefig(os.path.join(fig_dir, "time_vs_bottom_track_east_velocity.png"))

    # north velocity vs bin depth
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ds.bin_depth, ds.u_bm, 'o', markersize=1, color='purple')
    ax.set_title('North Velocity vs Bin Depth')
    ax.set_xlabel('Bin Depth')
    ax.set_ylabel('North Velocity')
    plt.savefig(os.path.join(fig_dir, "north_velocity_vs_bin_depth.png"))

    # time vs depth
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ds.datetime, ds.depth, 'o', markersize=1, color='brown')
    ax.set_title('Time vs Depth')
    ax.set_xlabel('Time')
    ax.set_ylabel('Depth')
    plt.savefig(os.path.join(fig_dir, "time_vs_depth.png"))

    # time vs backscatter
    fig, ax = plt.subplots(figsize=(10, 6))
    backscatter_mean = ds.backscatter.mean(dim='bin_depth')
    backscatter = backscatter_mean.mean(dim='beam')
    ax.plot(ds.datetime, backscatter, 'o', markersize=1, color='gray')
    ax.set_title('Time vs Backscatter')
    ax.set_xlabel('Time')
    ax.set_ylabel('Backscatter')
    plt.savefig(os.path.join(fig_dir, "time_vs_backscatter.png"))

    # time vs backscatter color plot with depth on the y axis and time on the x axis
    fig, ax = plt.subplots(figsize=(10, 6))
    backscatter_mean = ds.backscatter.mean(dim='beam')
    # plt.pcolormesh(ds.datetime.values, ds.bin_depth.values, backscatter_mean.values, cmap='viridis')
    plt.imshow(backscatter_mean.values, cmap='viridis', aspect='auto',extent=(ds.datetime.min().values, ds.datetime.max().values, 0, ds.bin_depth.max().values), origin='lower')
    ax.set_ylim(ds.bin_depth.max().values, 0)
    ax.set_xlim(ds.datetime.min().values, ds.datetime.max().values)
    ax.set_title('Time vs Backscatter')
    ax.set_xlabel('Time')
    ax.set_ylabel('Depth')
    plt.colorbar()
    plt.savefig(os.path.join(fig_dir, "time_vs_backscatter_color.png"))

    # time vs temperature
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ds.datetime, ds.temp, 'o', markersize=1, color='gray')
    ax.set_title('Time vs Temperature')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature')
    plt.savefig(os.path.join(fig_dir, "time_vs_temperature.png"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to NetCDF")
    parser.add_argument("file_path", type=str, help="Directory path of the CSV file")
    parser.add_argument("file_name", type=str, help="Name of the CSV file or all files in the directory if '--all' is specified")
    parser.add_argument("--rewrite", action="store_true", help="Rewrite existing NetCDF files if they already exist")
    args = parser.parse_args()

    if args.file_name.lower() == "--all":
        # Process all CSV files in the specified directory
        print("Processing all .MAT files in the specified directory...")
        # check if the provided path is a directory
        if not os.path.isdir(args.file_path):
            print(f"Error: {args.file_path} is not a valid directory.")
            sys.exit(1)
        # check what files have already been converted to netcdf
        for file in os.listdir(args.file_path):
            if file.endswith(".mat"):
                netcdf_file = file.replace(".mat", ".nc")
                if not os.path.exists(netcdf_file) or args.rewrite:
                    netcdf_file = convert_csv_to_netcdf(args.file_path, file)
                    print(f"Converted and Saved {file} to {netcdf_file}")
                    create_diagnostic_plots(netcdf_file)
                    print(f"Created diagnostic plots for {file}")
    else:
        # Process a single specified CSV file
        if not os.path.isfile(os.path.join(args.file_path, args.file_name)):
            print(f"Error: {args.file_name} is not a valid file.")
            sys.exit(1)
        print(f"Processing {args.file_name}...")
        netcdf_file = convert_csv_to_netcdf(args.file_path, args.file_name)
        print(f"Converted and Saved {args.file_name} to {netcdf_file}")
        create_diagnostic_plots(netcdf_file)
        print(f"Created diagnostic plots for {args.file_name}")

