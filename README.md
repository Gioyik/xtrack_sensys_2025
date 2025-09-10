# xTrack Sensys 2025 - Person Tracking Project

This project focuses on the identification, localization, and tracking of people using sensor data from the xTrack vehicle. The primary goal is to detect and track individuals, with a special focus on those wearing yellow safety vests, using data from an RGB-D camera and LiDAR.

## Features

- **Person Detection and Tracking:** Utilizes the YOLOv11 model from `ultralytics` to detect and track people in a video stream.
- **Advanced Re-Identification (Re-ID):** Implements two methods to ensure consistent tracking IDs, even when individuals leave and re-enter the frame:
  - **Custom Re-ID:** A custom implementation using appearance embeddings from a ResNet model to re-associate tracks.
  - **BoTSORT Re-ID:** Leverages the built-in re-identification capabilities of the BoTSORT tracker.
- **3D Localization:** Calculates the 3D position of detected individuals relative to the xTrack vehicle using RGB and depth data.
- **Safety Vest Detection:** Implements two methods for vest detection:
  - **Color-Based:** A simple and fast HSV color segmentation algorithm.
  - **Model-Based:** A more robust method using a pre-trained PyTorch model (MobileNetV2) for higher accuracy.
- **Selectable Trackers:** Allows for the selection of different tracking algorithms (`bytetrack` or `botsort`) for performance comparison.
- **Advanced LiDAR Processing:** Enhanced point cloud processing with noise filtering, ground plane removal using RANSAC, and clustering-based depth estimation for robust 3D localization.
- **Sensor Fusion:** Intelligent fusion of RGB-D camera and LiDAR data for improved accuracy, with distance-based weighting that favors depth camera for close objects and LiDAR for distant ones.
- **Frame Skipping:** Configurable frame skipping to improve processing speed while maintaining tracking continuity.
- **Cross-Platform GPU Support:** Full support for CUDA (NVIDIA), MPS (Apple Silicon), and CPU processing with automatic fallback.
- **Performance Benchmarking:** Provides tools to measure and log the performance (FPS, component-wise latency) of the tracking pipeline.
- **Real-time Data Logging:** Logs all tracking data for people with vests to a CSV file in real-time.
- **Robust Vest Detection:** Improved vest detection that works independently of 3D localization success, ensuring visual feedback even when positioning fails.
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

- `--benchmark`: Enables performance benchmarking.
  - When enabled, the script will print a summary of performance metrics upon completion, including average FPS and the time taken for key operations.

- `--vest_detection`: Specifies the vest detection method.
  - `color`: Use the original HSV color segmentation (default).
  - `model`: Use the advanced vest classifier.

- `--vest_model_path`: Specifies the path to the trained vest classifier model.
  - Default: `vest_model.pth`

- `--device`: Specifies the device to run the models on.
  - `cpu` | `cuda` | `mps`
  - Default: `cpu`
  - Note: `mps` provides GPU acceleration on Apple Silicon Macs

- `--localization_method`: Specifies the localization method to use.
  - `depth`: Use the RGB-D camera for 3D localization (default).
  - `lidar`: Use the LiDAR point cloud for 3D localization.
  - `fusion`: Intelligently combine both RGB-D and LiDAR data for optimal accuracy.

- `--jump_frames`: Number of frames to skip between processed frames for faster processing.
  - Default: `0` (no frame skipping)
  - Example: `--jump_frames 2` processes every 3rd frame (skips 2 frames between each processed frame)

- `--vest_threshold`: Yellow percentage threshold for vest detection.
  - Default: `5.0` (5%)
  - Range: `0.1` to `20.0`
  - Lower values = more sensitive detection, higher false positives
  - Higher values = more conservative detection, fewer false positives

- `--vest_persistence`: Number of consecutive frames vest must be detected before confirming.
  - Default: `1` (immediate detection)
  - Range: `1` to `10`
  - Higher values reduce false positives but may delay detection

- `--reid_threshold`: Similarity threshold for re-identifying lost tracks.
  - Default: `0.75` (conservative)
  - Range: `0.5` to `0.95`
  - Lower values = more re-identifications but higher chance of incorrect merging
  - Higher values = fewer re-identifications but better accuracy

- `--max_lost_frames`: Maximum frames a track can be lost before permanent deletion.
  - Default: `90` frames
  - Range: `30` to `300`
  - Higher values = longer memory but more computational overhead

### Example with Options

To run the script on the outdoor dataset using the `botsort` tracker with its native re-ID and basic debugging:
```bash
python3 src/person_tracker.py --dataset outdoor --tracker botsort --reid_method botsort --debug 1
```

To run the script and benchmark its performance:
```bash
python3 src/person_tracker.py --dataset outdoor --benchmark
```

To run the script using the model-based vest detector on a CUDA-enabled GPU:
```bash
python3 src/person_tracker.py --dataset outdoor --vest_detection model --device cuda --vest_model_path /path/to/your/vest_model.pth
```

To run the script using LiDAR-based localization:
```bash
python3 src/person_tracker.py --dataset outdoor --localization_method lidar
```

To run the script using sensor fusion (recommended for best accuracy):
```bash
python3 src/person_tracker.py --dataset outdoor --localization_method fusion
```

To run the script with frame skipping for faster processing:
```bash
python3 src/person_tracker.py --dataset outdoor --jump_frames 3 --benchmark
```

To run the script with MPS acceleration on Apple Silicon:
```bash
python3 src/person_tracker.py --dataset outdoor --device mps --reid_method custom
```

To run the script with optimized vest detection settings:
```bash
python3 src/person_tracker.py --dataset outdoor --vest_threshold 6.0 --vest_persistence 3 --debug 1
```

To run the script with improved ReID settings (recommended):
```bash
python3 src/person_tracker.py --dataset outdoor --reid_threshold 0.8 --max_lost_frames 60 --debug 1
```

To run the script with all optimized settings:
```bash
python3 src/person_tracker.py --dataset outdoor --vest_threshold 6.0 --vest_persistence 3 --reid_threshold 0.8 --debug 1 --benchmark
```

## Model-Based Vest Detection

When using the `--vest_detection model` option, the system relies on a pre-trained PyTorch model to classify whether a person is wearing a safety vest.

### Providing the Model

**You must provide your own trained model file.** The script expects a file named `vest_model.pth` by default, but you can specify a different path using the `--vest_model_path` argument.

### Model Architecture

The `src/vest_classifier.py` is built to load a `MobileNetV2` model from `torchvision`, with its final classification layer modified for binary classification (2 output classes: `no_vest` and `vest`).

If you are training your own model, you should follow this architecture. The classifier expects a model that has been trained on images of people with and without safety vests.

### Dependencies

Using the model-based detector requires `torch` and `torchvision`, which have been added to the `requirements.txt` file.

## Advanced LiDAR Processing and Sensor Fusion

The system implements state-of-the-art LiDAR processing and sensor fusion techniques for robust 3D localization:

### LiDAR Processing Pipeline

1. **Point Cloud Filtering:** Removes noise and outliers using statistical methods and distance-based filtering
2. **Ground Plane Removal:** Uses RANSAC algorithm to detect and remove ground plane points, focusing on above-ground objects
3. **Coordinate Transformation:** Transforms LiDAR points to camera frame using calibrated transformation matrices
4. **Frustum Culling:** Projects 3D points to 2D image plane and filters points within person bounding boxes
5. **Clustering-based Depth Estimation:** Uses DBSCAN clustering to group points and select the most reliable cluster for depth calculation
6. **Statistical Outlier Removal:** Applies IQR-based filtering as fallback for robust depth estimation

### Sensor Fusion Strategy

The `fusion` localization method intelligently combines RGB-D camera and LiDAR data:

- **Distance-based Weighting:** 
  - Objects < 5m: 70% depth camera, 30% LiDAR (depth camera more accurate for close range)
  - Objects ≥ 5m: 30% depth camera, 70% LiDAR (LiDAR more accurate for long range)
- **Automatic Fallback:** If one sensor fails, automatically uses the other
- **Validation:** Cross-validates measurements between sensors for increased reliability

### Performance Optimization

- **Frame Skipping:** `--jump_frames` allows processing every Nth frame for real-time performance
- **Device Optimization:** Automatic device selection with MPS support for Apple Silicon acceleration
- **Caching:** Efficient timestamp-based synchronization between sensors

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

## Troubleshooting

### Common Issues and Solutions

#### Poor LiDAR Localization Performance

**Symptoms:** LiDAR localization returns (0, 0, 0) or inaccurate positions.

**Solutions:**
1. Ensure LiDAR data has been properly converted using `scripts/os_pcap_to_pcd_csv.py`
2. Check that `timestamps.txt` file exists in the LiDAR output directory
3. Verify coordinate frame transformations in `coordinate_frames.py` are properly calibrated
4. Use sensor fusion (`--localization_method fusion`) for more robust results
5. Try adjusting filtering parameters in `lidar_utils.py` for your specific environment

#### Performance Issues

**Solutions:**
1. Use frame skipping: `--jump_frames 2` or higher values
2. Enable GPU acceleration: `--device cuda` or `--device mps` (Apple Silicon)
3. Use simplified tracking: `--tracker bytetrack` instead of `botsort`
4. Disable debugging: `--debug 0`

#### Multiple People Getting Same Track ID

**Problem:** Different people are incorrectly merged into the same track ID (e.g., all people become "Track ID 2").

**Symptoms:** 
- Frequent "Re-identified track X as Y" messages
- Multiple distinct people sharing the same ID
- Track IDs not increasing properly

**Solutions:**
1. **Increase ReID threshold**: `--reid_threshold 0.8` (default is 0.75)
2. **Reduce lost frame memory**: `--max_lost_frames 60` (default is 90)  
3. **Use BoTSORT ReID**: `--tracker botsort --reid_method botsort`
4. **Test conservative settings**: `--reid_threshold 0.85 --max_lost_frames 45`

#### Device Compatibility

**CUDA Issues:** If CUDA fails, the system automatically falls back to CPU with a warning.

**MPS Issues:** On older macOS versions, MPS may not be available. The system will fall back to CPU automatically.

**Memory Issues:** For large point clouds, consider reducing the max_distance parameter in the filtering functions.

## Processing Raw Data

The raw LIDAR data from the data directory needs to be converted into `.pcd` files and associated timestamps for synchronization. The script `scripts/os_pcap_to_pcd_csv.py` handles this conversion.

**Important:** The script has been updated to generate a `timestamps.txt` file, which is required for LiDAR-based localization. If you have previously processed your data, you must **re-run the conversion script** to generate this file.

To convert the outdoor dataset and save the output to a structured directory:

```bash
python3 scripts/os_pcap_to_pcd_csv.py data/data_outdoor/os_pcaps/ouster_20250604074152.pcap -o output
```

This will create the `output/ouster_20250604074152` directory, which will contain:
- `.pcd` files for each LiDAR scan.
- A `timestamps.txt` file containing the timestamp for each scan.
- An `_imu.csv` file with the raw IMU data.
