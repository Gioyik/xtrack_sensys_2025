# xTrack Sensor Fusion Project

This project aims to perform sensor fusion for object tracking using data from the xTrack vehicle.

## Project Structure

```
.
├── data/ # folder with provided ISIS data
│   ├── data_indoor/
│   └── data_outdoor/
├── output/
├── scripts/ # folder with dataset materials
├── src/
├── requirements.txt
└── README.md
```

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd xtrack_sensys_2025
    ```

2.  **Create and activate a Python virtual environment:**
    This project uses `pyenv` to manage Python versions and virtual environments.

    ```bash
    # Install pyenv if you don't have it
    # Follow instructions at https://github.com/pyenv/pyenv

    # Create a virtual environment with Python 3.10
    pyenv virtualenv 3.10 xtrack_env

    # Activate the virtual environment
    pyenv local xtrack_env
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Convert Raw Data:**
    The raw LIDAR data from the `data` directory needs to be converted into `.pcd` and `.csv` files. The script `scripts/os_pcap_to_pcd_csv.py` is used for this purpose.

    For example, to convert the outdoor dataset and save the output to the `output` directory:
    ```bash
    python3 scripts/os_pcap_to_pcd_csv.py data/data_outdoor/os_pcaps/ouster_20250604074152.pcap -o output/
    ```
    This will create the `output` directory if it doesn't exist and save the processed point clouds and IMU data there.
