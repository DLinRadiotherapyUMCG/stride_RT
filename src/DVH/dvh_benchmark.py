import numpy as np
import time

def benchmark_dvh_methods():
    """Compare efficiency of sorting once vs. multiple np.percentile calls"""
    
    # Simulate a large structure (e.g., whole lung with ~1M voxels)
    n_voxels = 60231 
    structure_doses = np.random.rand(n_voxels) * 70  # Random doses 0-70 Gy
    
    volume_points = [1, 2, 3, 4, 5] + list(range(10, 100, 5)) + [98, 99, 100]
    
    print(f"Testing with {n_voxels:,} voxels")
    print(f"Computing {len(volume_points)} D-values\n")
    
    # Method 1: Multiple np.percentile calls (current approach)
    start = time.time()
    for _ in range(10):  # Run 10 times for stable timing
        D_values_method1 = {}
        for volume_percent in volume_points:
            percentile = 100 - volume_percent
            D_values_method1[volume_percent] = np.percentile(structure_doses, percentile)
    time_method1 = time.time() - start
    
    # Method 2: Sort once, then index
    start = time.time()
    for _ in range(10):
        sorted_doses = np.sort(structure_doses)
        D_values_method2 = {}
        for volume_percent in volume_points:
            percentile = 100 - volume_percent
            index = int(len(sorted_doses) * percentile / 100)
            index = min(index, len(sorted_doses) - 1)  # Clamp to valid range
            D_values_method2[volume_percent] = sorted_doses[index]
    time_method2 = time.time() - start
    
    # Method 3: Sort once with interpolation (more accurate)
    start = time.time()
    for _ in range(10):
        sorted_doses = np.sort(structure_doses)
        D_values_method3 = {}
        for volume_percent in volume_points:
            percentile = 100 - volume_percent
            # Linear interpolation between indices
            float_index = len(sorted_doses) * percentile / 100
            lower_idx = int(np.floor(float_index))
            upper_idx = int(np.ceil(float_index))
            lower_idx = min(lower_idx, len(sorted_doses) - 1)
            upper_idx = min(upper_idx, len(sorted_doses) - 1)
            
            if lower_idx == upper_idx:
                D_values_method3[volume_percent] = sorted_doses[lower_idx]
            else:
                fraction = float_index - lower_idx
                D_values_method3[volume_percent] = (
                    sorted_doses[lower_idx] * (1 - fraction) + 
                    sorted_doses[upper_idx] * fraction
                )
    time_method3 = time.time() - start
    
    print(f"Method 1 (multiple np.percentile): {time_method1:.3f} sec")
    print(f"Method 2 (sort once, simple index): {time_method2:.3f} sec")
    print(f"Method 3 (sort once, interpolate):  {time_method3:.3f} sec")
    print(f"\nSpeedup vs Method 1:")
    print(f"  Method 2: {time_method1/time_method2:.1f}x faster")
    print(f"  Method 3: {time_method1/time_method3:.1f}x faster")
    
    # Verify they give similar results
    print(f"\nVerification (D95 values):")
    print(f"  Method 1: {D_values_method1[95]:.4f} Gy")
    print(f"  Method 2: {D_values_method2[95]:.4f} Gy")
    print(f"  Method 3: {D_values_method3[95]:.4f} Gy")

if __name__ == "__main__":
    benchmark_dvh_methods()