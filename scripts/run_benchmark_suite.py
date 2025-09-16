#!/usr/bin/env python3
"""
Automated benchmark suite for xTrack person tracking system.
Runs comprehensive tests across different configurations and collects results.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

def run_test(config, output_dir):
    """Run a single test configuration and capture results."""
    print(f"\n{'='*60}")
    print(f"Running Test: {config['name']}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        "python3", "src/person_tracker.py",
        "--benchmark",
        "--debug", "0"  # Minimize output for automation
    ]
    
    # Add configuration parameters
    for key, value in config["params"].items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run test and capture output
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        duration = time.time() - start_time
        
        # Save results
        test_result = {
            "name": config["name"],
            "config": config["params"],
            "duration": duration,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save individual result
        result_file = output_dir / f"{config['name'].replace(' ', '_')}.json"
        with open(result_file, 'w') as f:
            json.dump(test_result, f, indent=2)
        
        if result.returncode == 0:
            print(f"✅ Test completed successfully in {duration:.1f}s")
        else:
            print(f"❌ Test failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            
        return test_result
        
    except Exception as e:
        print(f"💥 Test crashed: {e}")
        return {
            "name": config["name"],
            "config": config["params"],
            "duration": time.time() - start_time,
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Run the complete benchmark suite."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("benchmark_results") / f"suite_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting xTrack Benchmark Suite")
    print(f"📁 Results will be saved to: {output_dir}")
    
    # Define test configurations
    test_configs = [
        {
            "name": "Baseline_Depth_Only",
            "params": {
                "dataset": "outdoor",
                "tracker": "bytetrack",
                "reid_method": "custom",
                "localization_method": "depth",
                "vest_detection": "color",
                "device": "cpu",
                "jump_frames": 0
            }
        },
        {
            "name": "LiDAR_Only",
            "params": {
                "dataset": "outdoor",
                "tracker": "bytetrack", 
                "reid_method": "custom",
                "localization_method": "lidar",
                "vest_detection": "color",
                "device": "cpu",
                "jump_frames": 0
            }
        },
        {
            "name": "Sensor_Fusion",
            "params": {
                "dataset": "outdoor",
                "tracker": "bytetrack",
                "reid_method": "custom", 
                "localization_method": "fusion",
                "vest_detection": "color",
                "device": "cpu",
                "jump_frames": 0
            }
        },
        {
            "name": "BoTSORT_Accurate",
            "params": {
                "dataset": "outdoor",
                "tracker": "botsort",
                "reid_method": "botsort",
                "localization_method": "fusion",
                "vest_detection": "color",
                "device": "cpu",
                "jump_frames": 0
            }
        },
        {
            "name": "Fast_Processing",
            "params": {
                "dataset": "outdoor",
                "tracker": "bytetrack",
                "reid_method": "custom",
                "localization_method": "depth",
                "vest_detection": "color",
                "device": "cpu",
                "jump_frames": 2
            }
        },
        {
            "name": "Maximum_Accuracy",
            "params": {
                "dataset": "outdoor",
                "tracker": "botsort",
                "reid_method": "botsort",
                "localization_method": "fusion",
                "vest_detection": "color",  # Would be "model" if vest_model.pth exists
                "device": "cpu",
                "jump_frames": 0
            }
        },
        # ReID Threshold Testing Matrix
        {
            "name": "ReID_Aggressive_0.6",
            "params": {
                "dataset": "outdoor",
                "tracker": "bytetrack",
                "reid_method": "custom",
                "reid_threshold": 0.6,
                "vest_detection": "color",
                "device": "cpu",
                "jump_frames": 0
            }
        },
        {
            "name": "ReID_Balanced_0.7",
            "params": {
                "dataset": "outdoor",
                "tracker": "bytetrack",
                "reid_method": "custom",
                "reid_threshold": 0.7,
                "vest_detection": "color",
                "device": "cpu",
                "jump_frames": 0
            }
        },
        {
            "name": "ReID_Conservative_0.8",
            "params": {
                "dataset": "outdoor",
                "tracker": "bytetrack",
                "reid_method": "custom",
                "reid_threshold": 0.8,
                "vest_detection": "color",
                "device": "cpu",
                "jump_frames": 0
            }
        },
        {
            "name": "ReID_Very_Conservative_0.85",
            "params": {
                "dataset": "outdoor",
                "tracker": "bytetrack",
                "reid_method": "custom",
                "reid_threshold": 0.85,
                "vest_detection": "color",
                "device": "cpu",
                "jump_frames": 0
            }
        }
    ]
    
    # Add GPU tests if available
    try:
        import torch
        if torch.cuda.is_available():
            test_configs.append({
                "name": "CUDA_Accelerated",
                "params": {
                    "dataset": "outdoor",
                    "tracker": "bytetrack",
                    "reid_method": "custom",
                    "localization_method": "fusion",
                    "vest_detection": "color",
                    "device": "cuda",
                    "jump_frames": 0
                }
            })
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            test_configs.append({
                "name": "MPS_Accelerated",
                "params": {
                    "dataset": "outdoor",
                    "tracker": "bytetrack",
                    "reid_method": "custom",
                    "localization_method": "fusion",
                    "vest_detection": "color",
                    "device": "mps",
                    "jump_frames": 0
                }
            })
    except ImportError:
        print("⚠️ PyTorch not available, skipping GPU tests")
    
    # Run all tests
    all_results = []
    start_time = time.time()
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n🔄 Running test {i}/{len(test_configs)}")
        result = run_test(config, output_dir)
        all_results.append(result)
        
        # Small delay between tests
        time.sleep(2)
    
    total_duration = time.time() - start_time
    
    # Generate summary report
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_duration": total_duration,
        "total_tests": len(test_configs),
        "successful_tests": sum(1 for r in all_results if r["success"]),
        "failed_tests": sum(1 for r in all_results if not r["success"]),
        "results": all_results
    }
    
    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"           BENCHMARK SUITE COMPLETE")
    print(f"{'='*60}")
    print(f"Total Duration: {total_duration/60:.1f} minutes")
    print(f"Tests Run: {len(test_configs)}")
    print(f"Successful: {summary['successful_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Results saved to: {output_dir}")
    
    # Print test results table
    print(f"\n📊 RESULTS SUMMARY")
    print("-" * 80)
    print(f"{'Test Name':<25} {'Status':<10} {'Duration':<12} {'Notes'}")
    print("-" * 80)
    
    for result in all_results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        duration = f"{result['duration']:.1f}s"
        notes = "OK" if result["success"] else "Check logs"
        print(f"{result['name']:<25} {status:<10} {duration:<12} {notes}")
    
    print("-" * 80)
    
    return summary['successful_tests'] == len(test_configs)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
