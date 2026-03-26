import numpy as np
from typing import Dict, List, Tuple, Optional

def compute_dvh_metrics(
    dose_array: np.ndarray,
    structure_masks: Dict[str, np.ndarray],
    step_size: int = 5,
    voxel_volume_cc: Optional[float] = None,
    dose_units: str = 'cGy'
) -> Dict[str, Dict]:
    """
    Compute comprehensive DVH metrics including V (volume) and D (dose) values.
    
    Parameters:
    -----------
    dose_array : np.ndarray
        3D array of radiation dose distribution (in Gy or cGy)
    structure_masks : Dict[str, np.ndarray]
        Dictionary mapping structure names to 3D binary masks
    dose_points : List[float], optional
        Dose levels for V calculations (Vx = volume receiving ≥x Gy)
        Default: [1, 2, 3, 4, 5, 10, 15, 20, ..., 95, 100]
    volume_points : List[float], optional
        Volume percentages for D calculations (Dx = dose to hottest x% of volume)
        Default: [1, 2, 3, 4, 5, 10, 15, 20, ..., 95, 98, 99, 100]
    voxel_volume_cc : float, optional
        Volume of a single voxel in cubic centimeters (cc). If provided, will convert volume from voxels to cc.
         If None, volume will be reported in number of voxels.
    
    Returns:
    --------
    Dict[str, Dict]
        {
            structure_name: {
                'V_values': {dose: volume_percent},
                'D_values': {volume_percent: dose},
                'statistics': {Dmax, Dmean, Dmin, volume_cc}
            }
        }
    
    Example:
    --------
    >>> results = compute_dvh_metrics(dose, masks)
    >>> print(f"PTV V50 = {results['PTV']['V_values'][50]:.1f}%")
    >>> print(f"PTV D95 = {results['PTV']['D_values'][95]:.2f} Gy")
    """

    if dose_units == 'cGy':
        dose_adjust_factor = 1   # convert Gy to cGy
    elif dose_units == 'Gy':
        dose_adjust_factor = 1 / 100.0  # convert cGy to Gy
    else:
        raise ValueError(f"Unsupported dose units: {dose_units}")
    
    # define the dose points for V calculations
    dose_points = [1, 2, 3, 4, 5] + list(range(10, 100+step_size, step_size))
    
    
    # Default volume points for D calculations
    volume_points = [1, 2, 3, 4, 5] + list(range(10, 100, step_size)) + [98, 99, 100]
    
    results_dict = {}
    
    for structure_name, mask in structure_masks.items():
        # Validate mask shape
        if mask.shape != dose_array.shape:
            raise ValueError(
                f"Mask for '{structure_name}' has shape {mask.shape}, but dose_array has shape {dose_array.shape}"
            )
        
        # Extract dose values within the structure
        structure_doses = dose_array[mask > 0]
        
        # Handle empty structures
        if len(structure_doses) == 0:
            # print(f"Warning: Structure '{structure_name}' has no voxels")

            for dose_level in dose_points:
                results_dict[f'{structure_name}_V{dose_level}'] = None
            for volume_percent in volume_points:
                results_dict[f'{structure_name}_D{volume_percent}'] = None

            results_dict[f'{structure_name}_Dmean'] = None
            results_dict[f'{structure_name}_Dmax'] = None
            results_dict[f'{structure_name}_Dmin'] = None
            results_dict[f'{structure_name}_volume'] = 0
            continue
        
        total_voxels = len(structure_doses)
        if voxel_volume_cc is not None:
            total_volume_cc = total_voxels * voxel_volume_cc
        else:
            total_volume_cc = total_voxels
        
        # ===== V VALUES: Volume receiving >= dose =====
        for dose_level in dose_points:
            voxels_above = np.sum(structure_doses >= dose_level)
            volume_percent = (voxels_above / total_voxels) * 100
            results_dict[f'{structure_name}_V{dose_level}'] = volume_percent
        
        # ===== D VALUES: Dose to hottest x% of volume =====
        for volume_percent in volume_points:
            # Calculate index corresponding to volume percentage
            # D95 = dose to the hottest 95% of volume
            # This is the 5th percentile of the dose distribution
            percentile = 100 - volume_percent
            dose_value = np.percentile(structure_doses, percentile)
            results_dict[f'{structure_name}_D{volume_percent}'] = dose_value

        
        # ===== MAIN STATISTICS =====
        results_dict[f'{structure_name}_Dmean'] = float(np.mean(structure_doses)) * dose_adjust_factor
        results_dict[f'{structure_name}_Dmax'] = float(np.max(structure_doses)) * dose_adjust_factor
        results_dict[f'{structure_name}_Dmin'] = float(np.min(structure_doses)) * dose_adjust_factor
        results_dict[f'{structure_name}_volume'] = total_volume_cc
    
    return results_dict


def print_dvh_summary(results: Dict[str, Dict], 
                      show_v_doses: List[float] = None,
                      show_d_volumes: List[float] = None):
    """
    Print a summary of DVH metrics in a readable format.
    
    Parameters:
    -----------
    results : Dict
        Output from compute_dvh_metrics
    show_v_doses : List[float], optional
        Specific V values to display (e.g., [5, 20, 50])
    show_d_volumes : List[float], optional
        Specific D values to display (e.g., [95, 50, 2])
    """
    
    for structure_name, metrics in results.items():
        print(f"\n{'='*60}")
        print(f"STRUCTURE: {structure_name}")
        print(f"{'='*60}")
        
        # Statistics
        stats = metrics['statistics']
        print(f"\nDose Statistics:")
        print(f"  Dmax  = {stats['Dmax']:.2f} Gy")
        print(f"  Dmean = {stats['Dmean']:.2f} Gy")
        print(f"  Dmin  = {stats['Dmin']:.2f} Gy")
        print(f"  Volume = {stats['volume_cc']} voxels")
        
        # V values
        print(f"\nV Values (Volume receiving ≥ dose):")
        v_vals = metrics['V_values']
        if show_v_doses:
            doses_to_show = [d for d in show_v_doses if d in v_vals]
        else:
            doses_to_show = sorted(v_vals.keys())[:10]  # Show first 10
        
        for dose in doses_to_show:
            print(f"  V{dose:>3.0f} = {v_vals[dose]:>6.2f}%")
        
        # D values
        print(f"\nD Values (Minimum dose to hottest x% of volume):")
        d_vals = metrics['D_values']
        if show_d_volumes:
            vols_to_show = [v for v in show_d_volumes if v in d_vals]
        else:
            vols_to_show = [98, 95, 50, 5, 2]  # Common clinical values
        
        for vol in vols_to_show:
            if vol in d_vals:
                print(f"  D{vol:>3.0f} = {d_vals[vol]:>6.2f} Gy")


# def export_dvh_full_csv(results: Dict[str, Dict], 
#                         filename: str = "dvh_full_results.csv"):
#     """
#     Export complete DVH results (V and D values) to CSV.
    
#     Parameters:
#     -----------
#     results : Dict
#         Output from compute_dvh_metrics
#     filename : str
#         Output CSV filename
#     """
#     import csv
    
#     with open(filename, 'w', newline='') as f:
#         writer = csv.writer(f)
        
#         # Header
#         writer.writerow(['Structure', 'Metric_Type', 'Parameter', 'Value', 'Unit'])
        
#         for structure_name, metrics in results.items():
#             # Statistics
#             stats = metrics['statistics']
#             writer.writerow([structure_name, 'Statistic', 'Dmax', stats['Dmax'], 'Gy'])
#             writer.writerow([structure_name, 'Statistic', 'Dmean', stats['Dmean'], 'Gy'])
#             writer.writerow([structure_name, 'Statistic', 'Dmin', stats['Dmin'], 'Gy'])
#             writer.writerow([structure_name, 'Statistic', 'Volume', stats['volume_cc'], 'voxels'])
            
#             # V values
#             for dose, volume_pct in sorted(metrics['V_values'].items()):
#                 writer.writerow([structure_name, 'V_value', f'V{dose}', volume_pct, '%'])
            
#             # D values
#             for vol_pct, dose in sorted(metrics['D_values'].items()):
#                 writer.writerow([structure_name, 'D_value', f'D{vol_pct}', dose, 'Gy'])
    
#     print(f"Full DVH results exported to {filename}")


# Example usage
if __name__ == "__main__":
    print("Creating example 3D dose distribution and structure masks...")
    
    # Example data
    np.random.seed(42)
    dose_array = np.random.rand(100, 100, 50) * 70  # 0-70 Gy
    
    structure_masks = {
        'PTV': np.random.randint(0, 2, (100, 100, 50)).astype(bool),
        'Heart': np.random.randint(0, 2, (100, 100, 50)).astype(bool),
    }
    
    # Compute all DVH metrics
    print("\nComputing DVH metrics (V and D values)...")
    results = compute_dvh_metrics(dose_array, structure_masks)

    print(results)
    
    # Display summary
    print_dvh_summary(
        results, 
        show_v_doses=[5, 20, 50, 70],
        show_d_volumes=[98, 95, 50, 5, 2]
    )
    
    # Export to CSV
    # export_dvh_full_csv(results, "dvh_complete.csv")
    
    # Example: Accessing specific values programmatically
    print("\n" + "="*60)
    print("PROGRAMMATIC ACCESS EXAMPLES")
    print("="*60)
    print(f"PTV V50 (volume receiving ≥50 Gy): {results['PTV']['V_values'][50]:.2f}%")
    print(f"PTV D95 (dose to hottest 95%): {results['PTV']['D_values'][95]:.2f} Gy")
    print(f"Heart Dmean: {results['Heart']['statistics']['Dmean']:.2f} Gy")
    print(f"Heart V5 (volume receiving ≥5 Gy): {results['Heart']['V_values'][5]:.2f}%")