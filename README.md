# xTrack Sensys 2025 - Person Tracking Project

This project focuses on the identification, localization, and tracking of people using sensor data from the xTrack vehicle. The primary goal is to detect and track individuals, with a special focus on those wearing safety vests, using data from an RGB-D camera.

## Features

- **Person Detection and Tracking:** Utilizes the YOLOv11 model from `ultralytics` to detect and track people in a video stream.
- **Advanced Re-Identification (Re-ID):** Implements two methods to ensure consistent tracking IDs, even when individuals leave and re-enter the frame:
  - **Custom Re-ID:** A custom implementation using appearance embeddings from a ResNet model to re-associate tracks.
  - **BoTSORT Re-ID:** Leverages the built-in re-identification capabilities of the BoTSORT tracker.
- **3D Localization:** Calculates the 3D position of detected individuals relative to the xTrack vehicle using RGB and depth data.
- **Safety Vest Detection:** Implements a color-based segmentation algorithm to identify people wearing yellow safety vests.
- **Selectable Trackers:** Allows for the selection of different tracking algorithms (`bytetrack` or `botsort`) for performance comparison.
- **Real-time Data Logging:** Logs all tracking data for people with vests to a CSV file in real-time.
- **Configurable Debugging:** Provides multiple levels of debugging output for development and fine-tuning.

## How to Run

The main script for this project is `src/person_tracker.py`. It can be run from the command line with several options to control its behavior.

### Prerequisites

Ensure you have a Python environment set up with all the required packages installed from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Basic Usage

To run the script with default settings (outdoor dataset, bytetrack tracker, custom re-ID, no debug output):
```bash
python3 src/person_tracker.py --dataset outdoor
```

### Command-Line Arguments

The script accepts the following command-line arguments:

- `--dataset`: Specifies the dataset to process.
  - `indoor` | `outdoor`
  - Default: `outdoor`

- `--tracker`: Specifies the tracking algorithm to use.
  - `bytetrack` | `botsort`
  - Default: `bytetrack`

- `--reid_method`: Specifies the re-identification method to use.
  - `custom`: Use the custom appearance-based re-ID.
  - `botsort`: Use the native BoTSORT re-ID. (Requires `--tracker botsort`).
  - Default: `custom`

- `--debug`: Sets the level of debugging output.
  - `0`: No debugging output (default).
  - `1`: Prints basic tracking and re-ID information to the console.
  - `2`: Prints detailed information and shows the vest detection mask for each person in a separate window.

### Example with Options

To run the script on the outdoor dataset using the `botsort` tracker with its native re-ID and basic debugging:
```bash
python3 src/person_tracker.py --dataset outdoor --tracker botsort --reid_method botsort --debug 1
```

## Output

- **Live Visualization:** A window will open showing the video stream with bounding boxes around detected people.
  - **Yellow Box:** Indicates a person detected with a safety vest.
  - **Red Box:** Indicates a person detected without a safety vest.
- **CSV Log File:** A file named `output/tracking_log_[dataset].csv` will be created, containing the timestamp, frame ID, object ID, and 3D position for each person detected *with a vest*.

## Fine-Tuning Vest Detection

The vest detection is performed in `src/vision.py` and is based on color segmentation in the HSV color space. If the detection is not accurate, you can fine-tune the following in the `detect_yellow_vest` function:
- `lower_yellow` and `upper_yellow`: These numpy arrays define the color range for yellow.
- The percentage threshold (currently `10.0`): This is the minimum percentage of yellow pixels required to classify a person as wearing a vest.

Using the `--debug 2` flag is highly recommended for this process, as it provides direct visual feedback on the performance of the color segmentation.

## Code Formatting and Linting

This project uses `ruff` for code formatting and linting. The configuration is defined in `ruff.toml`.

To format the code and fix any linting issues, run the following command from the project's root directory:
```bash
ruff format . && ruff check . --fix
```

## Processing Raw Data

The raw LIDAR data from the data directory needs to be converted into .pcd and .csv files. The script in `scripts/os_pcap_to_pcd_csv.py` is used for this purpose. It has been modified to add the option to setup a target folder.

To convert the outdoor dataset and save the output to the `output` folder:

```bash
python3 scripts/os_pcap_to_pcd_csv.py data/data_outdoor/os_pcaps/ouster_20250604074152.pcap -o output/
```

This will create the `output` directory if it doesn't exist and save the processed point clouds and IMU data there.
