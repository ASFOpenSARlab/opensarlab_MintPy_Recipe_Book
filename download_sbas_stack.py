from datetime import datetime
from pathlib import Path
import os
from math import ceil
from tqdm.auto import tqdm
from hyp3_sdk import Batch, HyP3
import opensarlab_lib as osl
import argparse
import zipfile

def migrate_sbas_stack(
    data_path, 
    new_stack, 
    hyp3_username, 
    hyp3_password, 
    hyp3_project_name, 
    product_type, 
    date_range, 
    flight_path
):
    """
    Migrate an SBAS stack using specified parameters.

    Parameters:
    - data_path: Path to store migrated data
    - new_stack: Whether to create a new data directory
    - hyp3_username: HyP3 username
    - hyp3_password: HyP3 password
    - hyp3_project_name: HyP3 project name
    - product_type: HyP3 product type, e.g., INSAR_GAMMA
    - date_range: Tuple of start and end dates (datetime.date objects)
    - flight_path: Sentinel-1 flight path
    """
    # Step 1: Set up the working directory
    data_path = Path(data_path).resolve()
    if not str(data_path).startswith("/work/CBI_InSAR"):
        raise ValueError("The data path must start with /work/CBI_InSAR.")
    
    if new_stack:
        if not data_path.exists():
            data_path.mkdir(parents=True)
            # run chmod to make sure the directory is accessible by all users in the group
            os.system(f"chmod -R g+rw {data_path}")
    elif not data_path.exists():
        raise ValueError("Specified data directory does not exist.")

    print(f"Using data path: {data_path}")

    # Step 2: Authenticate with HyP3
    hyp3 = HyP3(username=hyp3_username, password=hyp3_password)

    # Step 3: Retrieve project data
    if hyp3_project_name:
        print(f"Retrieving data for user: {hyp3_username}, project: {hyp3_project_name}")
        batch = hyp3.find_jobs(
            name=hyp3_project_name,
            job_type=product_type,
            user_id=hyp3_username
        ).filter_jobs(running=False, include_expired=False)
    else:
        print("No project name provided. Exiting.")
        return

    jobs = batch
    print(f"Originally there are {len(jobs)} products to migrate.")

    # Step 4: Filter by date range
    if date_range:
        print(f"Filtering jobs by date range: {date_range[0]} to {date_range[1]}")
        jobs = osl.filter_jobs_by_date(jobs, date_range)
    print(f"After filtering date: there are {len(jobs)} products to migrate.")
    osl.set_paths_orbits(jobs)
    paths = set()
    for p in jobs:
        paths.add(p.path)
    print(f"Available flight paths: {paths}")
    if flight_path and flight_path not in paths:
        print(f"Invalid flight path: {flight_path}. Exiting.")
        return
    elif not flight_path:
        flight_path = paths    
    # Step 5: Filter by flight path
    if flight_path:
        jobs = osl.filter_jobs_by_path(jobs, [flight_path])
    print(f"There are {len(jobs)} products to migrate.")

    # Step 6: Download and process data
    print(f"Downloading and processing {len(jobs)} products...")
    # Retrieve SLURM environment variables
    task_id = int(os.getenv('SLURM_PROCID', 0))  # Task ID (unique for each task)
    num_tasks = int(os.getenv('SLURM_NTASKS', 1))  # Total number of tasks

    # Distribute jobs across tasks
    total_jobs = len(jobs)
    chunk_size = ceil(total_jobs / num_tasks)
    start_index = task_id * chunk_size
    end_index = min(start_index + chunk_size, total_jobs)

    # Each task processes its subset of jobs
    task_jobs = jobs[start_index:end_index]

    print(f"Task {task_id}: Processing jobs {start_index} to {end_index}")

    for job in task_jobs:
        for file in job.files:
            filename = file['filename']
            filepath = os.path.join(data_path, filename)
            if os.path.exists(filepath):
                print(f"Task {task_id}: {filename} already exists. Skipping.")
            else:
                print(f"Task {task_id}: Downloading {filename}...")
                job.download_files(data_path)
                print(f"Task {task_id}: Downloaded {filename}...")

    print(f"Download complete. Data stored in: {data_path}")

    while(True):
        # List all zip files in the data path
        all_files = sorted([f for f in data_path.iterdir() if f.suffix == '.zip'], key=lambda f: f.stat().st_ctime)
        # Wait for all files to download
        if len(all_files) < len(jobs):
            print(f"Task {task_id}: Waiting for all files to download...")
            continue
        else:
            # Distribute files across tasks
            total_files = len(all_files)
            chunk_size = ceil(total_files / num_tasks)
            start_index = task_id * chunk_size
            end_index = min(start_index + chunk_size, total_files)

            # Each task processes its subset of files
            task_files = all_files[start_index:end_index]
            print(f"Task {task_id}: Processing files {start_index} to {end_index}")
            # Unzip each file in the task's subset
            for zip_file in task_files:
                output_dir = data_path / zip_file.stem  # Directory named after the zip file
                if output_dir.exists():
                    print(f"Task {task_id}: {output_dir} already exists. Skipping.")
                else:
                    print(f"Task {task_id}: Unzipping {zip_file}...")
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(data_path)
                    os.system(f"chmod -R g+rw {output_dir}")
                    print(f"Task {task_id}: Unzipped {zip_file} to {output_dir}.")

            print("All tasks complete. Exiting.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate SBAS stack data")
    parser.add_argument("--data-path", required=True, help="Path to store migrated data")
    parser.add_argument("--new-stack", type=bool, default=True, help="Create a new data directory (default: True)")
    parser.add_argument("--hyp3-username", required=True, help="HyP3 username")
    parser.add_argument("--hyp3-password", required=True, help="HyP3 password")
    parser.add_argument("--hyp3-project-name", required=True, help="HyP3 project name")
    parser.add_argument("--product-type", default="INSAR_GAMMA", help="HyP3 product type (default: INSAR_GAMMA)")
    parser.add_argument("--date-range", nargs=2, type=str, help="Start and end dates in YYYY-MM-DD format")
    parser.add_argument("--flight-path", help="Sentinel-1 flight path")
    
    args = parser.parse_args()

    # Parse date range into datetime objects
    date_range = None
    if args.date_range:
        date_range = (datetime.strptime(args.date_range[0], "%Y-%m-%d").date(),
                      datetime.strptime(args.date_range[1], "%Y-%m-%d").date())
    # convert flight_path to int
    if args.flight_path:
        args.flight_path = int(args.flight_path)

    migrate_sbas_stack(
        data_path=args.data_path,
        new_stack=args.new_stack,
        hyp3_username=args.hyp3_username,
        hyp3_password=args.hyp3_password,
        hyp3_project_name=args.hyp3_project_name,
        product_type=args.product_type,
        date_range=date_range,
        flight_path=args.flight_path,
    )
