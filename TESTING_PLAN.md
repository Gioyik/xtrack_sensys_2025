# xTrack Testing and Benchmarking Plan

## Overview

This document outlines comprehensive testing procedures to evaluate the xTrack person tracking system across different configurations and identify optimal settings for various scenarios.

## Testing Objectives

1. **Performance Benchmarking**: Measure FPS, latency, and resource usage
2. **Accuracy Evaluation**: Assess tracking precision, vest detection accuracy, and 3D localization quality
3. **Robustness Testing**: Evaluate system stability under different conditions
4. **Configuration Optimization**: Identify best parameter combinations for different use cases

## Test Datasets

- **Indoor Dataset**: `data_indoor` - Controlled environment, close-range interactions
- **Outdoor Dataset**: `data_outdoor` - Natural lighting, varied distances, more challenging

## Testing Matrix

### 1. Localization Method Comparison

Test all three localization methods to compare accuracy and performance:

```bash
# Depth Camera Only (Baseline)
python3 src/person_tracker.py --dataset outdoor --localization_method depth --debug 1 --benchmark

# LiDAR Only 
python3 src/person_tracker.py --dataset outdoor --localization_method lidar --debug 1 --benchmark

# Sensor Fusion (Expected Best)
python3 src/person_tracker.py --dataset outdoor --localization_method fusion --debug 1 --benchmark
```

### 2. Tracker Algorithm Comparison

Compare different tracking algorithms:

```bash
# ByteTrack (Fast)
python3 src/person_tracker.py --dataset outdoor --tracker bytetrack --reid_method custom --debug 1 --benchmark

# BoTSORT (Accurate)
python3 src/person_tracker.py --dataset outdoor --tracker botsort --reid_method botsort --debug 1 --benchmark
```

### 3. Vest Detection Method Comparison

Test both vest detection approaches:

```bash
# Color-based Detection (Fast)
python3 src/person_tracker.py --dataset outdoor --vest_detection color --debug 2 --benchmark

# Model-based Detection (Accurate - requires trained model)
python3 src/person_tracker.py --dataset outdoor --vest_detection model --vest_model_path vest_model.pth --debug 2 --benchmark
```

### 4. Device Performance Testing

Test across different compute devices:

```bash
# CPU Baseline
python3 src/person_tracker.py --dataset outdoor --device cpu --benchmark

# CUDA (if available)
python3 src/person_tracker.py --dataset outdoor --device cuda --benchmark

# MPS (Apple Silicon)
python3 src/person_tracker.py --dataset outdoor --device mps --benchmark
```

### 5. Frame Skipping Performance Analysis

Test impact of frame skipping on performance vs accuracy:

```bash
# No skipping (baseline)
python3 src/person_tracker.py --dataset outdoor --jump_frames 0 --benchmark

# Skip 1 frame (process every 2nd frame)
python3 src/person_tracker.py --dataset outdoor --jump_frames 1 --benchmark

# Skip 2 frames (process every 3rd frame)
python3 src/person_tracker.py --dataset outdoor --jump_frames 2 --benchmark

# Skip 4 frames (process every 5th frame)
python3 src/person_tracker.py --dataset outdoor --jump_frames 4 --benchmark

# Skip 9 frames (process every 10th frame)
python3 src/person_tracker.py --dataset outdoor --jump_frames 9 --benchmark
```

### 6. Vest Detection Threshold Optimization

Test different vest detection thresholds to find optimal balance:

```bash
# Very sensitive (3% threshold)
python3 src/person_tracker.py --dataset outdoor --vest_detection color --vest_threshold 3.0 --debug 2 --benchmark

# Slightly sensitive (4% threshold)
python3 src/person_tracker.py --dataset outdoor --vest_detection color --vest_threshold 4.0 --debug 2 --benchmark

# Default (5% threshold) 
python3 src/person_tracker.py --dataset outdoor --vest_detection color --vest_threshold 5.0 --debug 2 --benchmark

# Recommended based on analysis (6% threshold)
python3 src/person_tracker.py --dataset outdoor --vest_detection color --vest_threshold 6.0 --debug 2 --benchmark

# Conservative (8% threshold)
python3 src/person_tracker.py --dataset outdoor --vest_detection color --vest_threshold 8.0 --debug 2 --benchmark

# Very conservative (12% threshold)
python3 src/person_tracker.py --dataset outdoor --vest_detection color --vest_threshold 12.0 --debug 2 --benchmark
```

### 6b. Temporal Persistence Testing

Test different persistence requirements to reduce false positives:

```bash
# No persistence (immediate detection)
python3 src/person_tracker.py --dataset outdoor --vest_persistence 1 --debug 1 --benchmark

# Require 2 consecutive frames
python3 src/person_tracker.py --dataset outdoor --vest_persistence 2 --debug 1 --benchmark

# Require 3 consecutive frames (recommended for noise reduction)
python3 src/person_tracker.py --dataset outdoor --vest_persistence 3 --debug 1 --benchmark

# Require 5 consecutive frames (very conservative)
python3 src/person_tracker.py --dataset outdoor --vest_persistence 5 --debug 1 --benchmark
```

### 6c. ReID Threshold Optimization (Critical for Track Assignment)

**IMPORTANT**: This is crucial for preventing incorrect track ID merging. Test different ReID similarity thresholds to find the optimal balance between track continuity and identity accuracy.

#### Visual Inspection Focus:
- Count unique track IDs appearing in the video
- Monitor "Re-identified track X as Y" messages frequency
- Verify each person maintains consistent track ID
- Check for false re-identifications (different people merged)

```bash
# Very aggressive ReID (high false positives - many incorrect merges)
python3 src/person_tracker.py --dataset outdoor --reid_threshold 0.5 --debug 1 --benchmark

# Moderately aggressive (some false positives expected)
python3 src/person_tracker.py --dataset outdoor --reid_threshold 0.6 --debug 1 --benchmark

# Balanced approach (some merging, generally accurate)
python3 src/person_tracker.py --dataset outdoor --reid_threshold 0.7 --debug 1 --benchmark

# Default conservative (recommended baseline)
python3 src/person_tracker.py --dataset outdoor --reid_threshold 0.75 --debug 1 --benchmark

# More conservative (fewer re-IDs, better accuracy)
python3 src/person_tracker.py --dataset outdoor --reid_threshold 0.8 --debug 1 --benchmark

# Very conservative (minimal re-ID, highest accuracy)
python3 src/person_tracker.py --dataset outdoor --reid_threshold 0.85 --debug 1 --benchmark

# Extremely conservative (almost no re-ID, many lost tracks)
python3 src/person_tracker.py --dataset outdoor --reid_threshold 0.9 --debug 1 --benchmark
```

#### Combined ReID and Memory Testing:

```bash
# Short memory + conservative threshold (quick forgetting, accurate re-ID)
python3 src/person_tracker.py --dataset outdoor --reid_threshold 0.8 --max_lost_frames 30 --debug 1 --benchmark

# Long memory + very conservative threshold (long retention, minimal re-ID)
python3 src/person_tracker.py --dataset outdoor --reid_threshold 0.85 --max_lost_frames 120 --debug 1 --benchmark

# Balanced memory + balanced threshold
python3 src/person_tracker.py --dataset outdoor --reid_threshold 0.75 --max_lost_frames 60 --debug 1 --benchmark

# Quick memory cycling (for high-density scenarios)
python3 src/person_tracker.py --dataset outdoor --reid_threshold 0.8 --max_lost_frames 20 --debug 1 --benchmark
```

#### ReID Quality Assessment Metrics:

For each test, manually evaluate:

1. **Track ID Count**: Total unique track IDs generated
2. **Re-ID Frequency**: Number of "Re-identified track X as Y" messages
3. **False Merge Rate**: Visual count of different people sharing same ID
4. **Track Fragmentation**: Same person getting multiple different IDs
5. **Track Persistence**: How long tracks survive occlusions/exits

#### Expected Results by Threshold:

| Threshold | Re-ID Frequency | False Merges | Track Fragmentation | Best Use Case |
|-----------|----------------|--------------|-------------------|---------------|
| 0.5       | Very High      | Many         | Low               | Never use - too aggressive |
| 0.6       | High           | Some         | Low               | Dense crowds (with caution) |
| 0.7       | Moderate       | Few          | Moderate          | Balanced scenarios |
| 0.75      | Low            | Very Few     | Moderate          | **Recommended default** |
| 0.8       | Very Low       | Rare         | Higher            | High accuracy priority |
| 0.85      | Minimal        | Almost None  | High              | Forensic analysis |
| 0.9       | Almost None    | None         | Very High         | Research/validation only |

#### Visual Benchmarking Checklist:

**🔍 What to Watch For:**

✅ **Good Signs:**
- Each person maintains consistent color box throughout appearance
- Track IDs increment properly (1, 2, 3, 4...)
- Re-ID messages only when person genuinely re-enters frame
- Stable track IDs during normal movement

❌ **Problem Signs:**
- Multiple people with same track ID simultaneously
- Frequent re-ID messages (every few frames)
- Track IDs jumping between people
- All people eventually becoming "Track ID 2"

**📊 Benchmark Data Collection:**

For systematic evaluation, record:
```
Threshold: X.XX
Total Runtime: XXX seconds
Unique Track IDs: XX
Re-ID Messages: XX
Visual False Merges: XX
Longest Track Duration: XX frames
Shortest Track Duration: XX frames
```

**🎯 Optimal Threshold Selection:**

Choose threshold where:
- False merges < 5% of total tracks
- Re-ID messages < 10% of total detections  
- Visual inspection shows good separation
- Track persistence matches scene complexity

### 7. Comprehensive Configuration Matrix

Test optimal combinations for different scenarios:

```bash
# Real-time Performance Setup
python3 src/person_tracker.py --dataset outdoor --tracker bytetrack --reid_method custom --localization_method depth --vest_detection color --jump_frames 2 --device mps --benchmark

# Maximum Accuracy Setup
python3 src/person_tracker.py --dataset outdoor --tracker botsort --reid_method botsort --localization_method fusion --vest_detection model --jump_frames 0 --device mps --debug 1 --benchmark

# Balanced Setup
python3 src/person_tracker.py --dataset outdoor --tracker bytetrack --reid_method custom --localization_method fusion --vest_detection color --jump_frames 1 --device mps --benchmark

# Long-range Optimized Setup
python3 src/person_tracker.py --dataset outdoor --tracker botsort --reid_method botsort --localization_method lidar --vest_detection color --jump_frames 1 --device mps --benchmark
```

### 8. Indoor vs Outdoor Comparison

Run identical configurations on both datasets:

```bash
# Indoor testing
python3 src/person_tracker.py --dataset indoor --tracker bytetrack --localization_method fusion --benchmark

# Outdoor testing  
python3 src/person_tracker.py --dataset outdoor --tracker bytetrack --localization_method fusion --benchmark
```

## Expected Results Analysis

### Performance Metrics to Track

1. **FPS (Frames Per Second)**
   - Target: >10 FPS for real-time applications
   - Expected order: depth > lidar > fusion (processing time)

2. **Component Latency**
   - YOLO inference time
   - ReID processing time 
   - Vest detection time
   - 3D localization time

3. **Memory Usage**
   - Peak memory consumption
   - GPU memory utilization

### Accuracy Metrics to Evaluate

1. **Vest Detection Quality**
   - False positive rate (non-vest people marked as wearing vests)
   - False negative rate (vest-wearing people not detected)
   - Threshold optimization curve

2. **3D Localization Accuracy**
   - Position stability (variance over time for stationary objects)
   - Cross-sensor consistency (depth vs LiDAR vs fusion)
   - Distance estimation accuracy

3. **Tracking Continuity**
   - Track persistence (how long tracks are maintained)
   - ID switches (how often track IDs change incorrectly)
   - ReID success rate

## Automated Testing Script

Create a test automation script that runs all configurations and collects results:

```bash
# Create automated test runner
python3 scripts/run_benchmark_suite.py --output_dir benchmark_results/
```

## Results Interpretation Guidelines

### Vest Detection Threshold Analysis

Based on your output sample:

- **Current 5% threshold**: Shows borderline cases around 6-7%
- **Recommended testing**: 3%, 5%, 8%, 12% thresholds
- **Expected findings**: 
  - Lower thresholds: More detections but higher false positives
  - Higher thresholds: Fewer false positives but may miss actual vests

### Performance vs Accuracy Trade-offs

1. **Real-time Applications**: Prioritize FPS, accept some accuracy loss
2. **Forensic Analysis**: Prioritize accuracy, processing time less critical  
3. **Edge Deployment**: Balance accuracy with resource constraints

### Configuration Recommendations

Based on expected results:

- **Close-range indoor**: Depth camera sufficient, higher FPS
- **Long-range outdoor**: LiDAR or fusion necessary
- **Mixed scenarios**: Fusion provides best overall performance
- **Resource-constrained**: Frame skipping with optimized thresholds

## Test Environment Requirements

- Python environment: `xtrack_env` (already set up)
- Sufficient disk space for benchmark logs
- GPU access for device comparison tests
- Timing measurement tools for performance analysis

## Success Criteria

- Achieve >15 FPS on target hardware for real-time config
- <5% false positive rate for vest detection
- <10% variance in 3D position for stationary objects
- Successful processing of complete datasets without crashes
