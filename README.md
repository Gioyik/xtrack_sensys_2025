# xTrack Sensys 2025 - Person Tracking Project

This project focuses on the identification, localization, and tracking of people using sensor data from the xTrack vehicle. The primary goal is to detect and track individuals, with a special focus on those wearing safety vests, using data from an RGB-D camera.

## Features

- **Person Detection and Tracking:** Utilizes the YOLOv11 model from `ultralytics` to detect and track people in a video stream.
- **3D Localization:** Calculates the 3D position of detected individuals relative to the xTrack vehicle using RGB and depth data.
- **Safety Vest Detection:** Implements a color-based segmentation algorithm to identify people wearing yellow safety vests.
- **Selectable Trackers:** Allows for the selection of different tracking algorithms (`bytetrack` or `botsort`) for performance comparison.
- **Real-time Data Logging:** Logs all tracking data for people with vests to a CSV file in real-time.
- **Configurable Debugging:** Provides multiple levels of debugging output for development and fine-tuning.

### Fine-Tuning Vest Detection

The vest detection is performed in `src/vision.py` and is based on color segmentation in the HSV color space. If the detection is not accurate, you can fine-tune the following in the `detect_yellow_vest` function:
- `lower_yellow` and `upper_yellow`: These numpy arrays define the color range for yellow.
- The percentage threshold (currently `10.0`): This is the minimum percentage of yellow pixels required to classify a person as wearing a vest.

Using the `--debug 2` flag is highly recommended for this process, as it provides direct visual feedback on the performance of the color segmentation.

## Usage

Ensure you have a Python environment set up with all the required packages installed from `requirements.txt`.

```bash
pip install -r requirements.txt
```

The main script for this project is `src/person_tracker.py`. It can be run from the command line with several options to control its behavior. To run the script with default settings (outdoor dataset, bytetrack tracker, no debug output):

```bash
python3 src/person_tracker.py --dataset outdoor
```

This tool will start:

- **Live Visualization:** A window will open showing the video stream with bounding boxes around detected people.
  - **Yellow Box:** Indicates a person detected with a safety vest.
  - **Red Box:** Indicates a person detected without a safety vest.
- **CSV Log File:** A file named `output/tracking_log_[dataset].csv` will be created, containing the timestamp, frame ID, object ID, and 3D position for each person detected *with a vest*.

### Command-Line Arguments

The script accepts the following command-line arguments:

- `--dataset`: Specifies the dataset to process.
  - `indoor`: Use the indoor video and data.
  - `outdoor`: Use the outdoor video and data.
  - Default: `outdoor`

- `--tracker`: Specifies the tracking algorithm to use.
  - `bytetrack`: Use the ByteTrack algorithm.
  - `botsort`: Use the BoT-SORT algorithm.
  - Default: `bytetrack`

- `--debug`: Sets the level of debugging output.
  - `0`: No debugging output (default).
  - `1`: Prints basic tracking information to the console.
  - `2`: Prints detailed information and shows the vest detection mask for each person in a separate window.

To run the script on the outdoor dataset using the `botsort` tracker and full debugging visualization:

```bash
python3 src/person_tracker.py --dataset outdoor --tracker botsort --debug 2
```

## Processing Raw Data

The raw LIDAR data from the data directory needs to be converted into .pcd and .csv files. The script in `scripts/os_pcap_to_pcd_csv.py` is used for this purpose. It has been modified to add the option to setup a target folder.

To convert the outdoor dataset and save the output to the `output` folder:

```bash
python3 scripts/os_pcap_to_pcd_csv.py data/data_outdoor/os_pcaps/ouster_20250604074152.pcap -o output/
```

This will create the `output` directory if it doesn't exist and save the processed point clouds and IMU data there.