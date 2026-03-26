"""
Diagnostic script to check if RTSTRUCT matches CT series
This helps identify UID mismatch issues before processing
"""

import os
import numpy as np
from pydicom import dcmread
from glob import glob
from collections import defaultdict


def analyze_rtstruct_ct_match(ct_folder, rtstruct_path):
    """
    Analyze if an RTSTRUCT matches a CT series and report detailed findings
    
    Parameters:
    -----------
    ct_folder : str
        Path to folder containing CT DICOM files
    rtstruct_path : str
        Path to RTSTRUCT DICOM file
    
    Returns:
    --------
    dict : Analysis results with matching statistics
    """
    print("="*80)
    print("RTSTRUCT-CT MATCHING ANALYSIS")
    print("="*80)
    
    # Load RTSTRUCT
    print(f"\n1. Loading RTSTRUCT: {os.path.basename(rtstruct_path)}")
    ds_rtstruct = dcmread(rtstruct_path)
    
    # Load CT series
    print(f"2. Loading CT series: {ct_folder}")
    ct_files = sorted(glob(os.path.join(ct_folder, "*.dcm")))
    print(f"   Found {len(ct_files)} CT slices")
    
    ct_slices = []
    for ct_file in ct_files:
        try:
            ct_slices.append(dcmread(ct_file))
        except:
            pass
    
    if not ct_slices:
        print("   ERROR: No valid CT slices found!")
        return None
    
    # Check Patient ID
    print("\n3. Checking Patient ID:")
    rtstruct_patient = ds_rtstruct.PatientID
    ct_patient = ct_slices[0].PatientID
    print(f"   RTSTRUCT Patient ID: {rtstruct_patient}")
    print(f"   CT Patient ID:       {ct_patient}")
    patient_match = (rtstruct_patient == ct_patient)
    print(f"   Match: {'✓ YES' if patient_match else '✗ NO'}")
    
    # Check Frame of Reference UID
    print("\n4. Checking Frame of Reference UID:")
    rtstruct_frame_uid = ds_rtstruct.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
    ct_frame_uid = ct_slices[0].FrameOfReferenceUID
    print(f"   RTSTRUCT FoR UID: {rtstruct_frame_uid}")
    print(f"   CT FoR UID:       {ct_frame_uid}")
    frame_match = (rtstruct_frame_uid == ct_frame_uid)
    print(f"   Match: {'✓ YES' if frame_match else '✗ NO'}")
    
    # Extract referenced UIDs from RTSTRUCT
    print("\n5. Extracting referenced slice UIDs from RTSTRUCT:")
    referenced_uids = set()
    contour_count = 0
    
    for roi_contour in ds_rtstruct.ROIContourSequence:
        if not hasattr(roi_contour, 'ContourSequence'):
            continue
        
        for contour in roi_contour.ContourSequence:
            contour_count += 1
            if hasattr(contour, 'ContourImageSequence'):
                for img_ref in contour.ContourImageSequence:
                    referenced_uids.add(img_ref.ReferencedSOPInstanceUID)
    
    print(f"   Found {len(referenced_uids)} unique referenced slice UIDs")
    print(f"   Total contours: {contour_count}")
    
    # Get CT slice UIDs
    print("\n6. Extracting CT slice UIDs:")
    ct_uids = {s.SOPInstanceUID for s in ct_slices}
    print(f"   CT series has {len(ct_uids)} unique slice UIDs")
    
    # Check UID overlap
    print("\n7. Checking UID overlap:")
    matching_uids = referenced_uids.intersection(ct_uids)
    match_percentage = (len(matching_uids) / len(referenced_uids) * 100) if referenced_uids else 0
    
    print(f"   Matching UIDs: {len(matching_uids)} / {len(referenced_uids)} ({match_percentage:.1f}%)")
    
    if match_percentage == 0:
        print("   ✗ CRITICAL: No matching UIDs - Wrong CT series or different acquisition!")
    elif match_percentage < 50:
        print("   ⚠ WARNING: Less than 50% match - Partial mismatch")
    elif match_percentage < 100:
        print(f"   ⚠ WARNING: {100-match_percentage:.1f}% of slices don't match")
    else:
        print("   ✓ SUCCESS: All UIDs match perfectly!")
    
    # Check Z-position matching (spatial matching)
    print("\n8. Checking Z-position (spatial) matching:")
    
    # Build Z-position maps
    ct_z_positions = {}
    for ct_slice in ct_slices:
        z_pos = round(float(ct_slice.ImagePositionPatient[2]), 1)
        ct_z_positions[z_pos] = ct_slice
    
    rtstruct_z_positions = set()
    z_match_count = 0
    z_close_match_count = 0
    
    for roi_contour in ds_rtstruct.ROIContourSequence:
        if not hasattr(roi_contour, 'ContourSequence'):
            continue
        
        for contour in roi_contour.ContourSequence:
            try:
                contour_data = np.array(contour.ContourData).reshape(-1, 3)
                contour_z = round(float(contour_data[0, 2]), 1)
                rtstruct_z_positions.add(contour_z)
                
                # Check for exact match
                if contour_z in ct_z_positions:
                    z_match_count += 1
                else:
                    # Check for close match (within 1mm)
                    closest_z = min(ct_z_positions.keys(), key=lambda z: abs(z - contour_z))
                    if abs(closest_z - contour_z) < 1.0:
                        z_close_match_count += 1
            except:
                pass
    
    z_match_percentage = (z_match_count / contour_count * 100) if contour_count else 0
    z_close_percentage = ((z_match_count + z_close_match_count) / contour_count * 100) if contour_count else 0
    
    print(f"   CT Z-range: {min(ct_z_positions.keys()):.1f} to {max(ct_z_positions.keys()):.1f} mm")
    print(f"   RTSTRUCT Z-range: {min(rtstruct_z_positions):.1f} to {max(rtstruct_z_positions):.1f} mm")
    print(f"   Exact Z-matches: {z_match_count} / {contour_count} ({z_match_percentage:.1f}%)")
    print(f"   Close Z-matches (<1mm): {z_match_count + z_close_match_count} / {contour_count} ({z_close_percentage:.1f}%)")
    
    # Final assessment
    print("\n" + "="*80)
    print("FINAL ASSESSMENT:")
    print("="*80)
    
    if match_percentage == 100:
        print("✓ PERFECT MATCH: UIDs match perfectly. No fixing needed.")
        recommendation = "USE_DIRECTLY"
    elif z_match_percentage > 90:
        print("⚠ UID MISMATCH BUT SPATIAL MATCH: UIDs don't match but geometry is correct.")
        print("  → This is the issue you're experiencing!")
        print("  → Use fix_missing_uids=True in RTStructBuilder.create_from()")
        recommendation = "FIX_UIDS"
    elif z_close_percentage > 90:
        print("⚠ CLOSE SPATIAL MATCH: Geometry is similar but not exact.")
        print("  → May work with UID fixing but verify results carefully")
        recommendation = "FIX_UIDS_WITH_CAUTION"
    else:
        print("✗ MISMATCH: This RTSTRUCT doesn't match this CT series!")
        print("  → Find the correct CT series that was used to create this RTSTRUCT")
        recommendation = "WRONG_CT_SERIES"
    
    print("\n")
    
    # Return results
    results = {
        'patient_match': patient_match,
        'frame_match': frame_match,
        'uid_match_percentage': match_percentage,
        'z_match_percentage': z_match_percentage,
        'z_close_match_percentage': z_close_percentage,
        'recommendation': recommendation,
        'matching_uids': len(matching_uids),
        'total_referenced_uids': len(referenced_uids),
        'total_ct_uids': len(ct_uids),
        'total_contours': contour_count
    }
    
    return results


def check_all_patients(base_data_dir, output_csv=None):
    """
    Check all patients in a directory structure
    
    Parameters:
    -----------
    base_data_dir : str
        Base directory containing patient folders
    output_csv : str, optional
        Path to save results as CSV
    """
    import pandas as pd
    
    results_list = []
    
    patient_folders = [f for f in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, f))]
    
    for patient_id in patient_folders:
        patient_path = os.path.join(base_data_dir, patient_id)
        
        # Find CT folder
        ct_folder = None
        rtstruct_file = None
        
        for root, dirs, files in os.walk(patient_path):
            if 'CT' in dirs:
                ct_folder = os.path.join(root, 'CT')
            if 'RTSTRUCT' in dirs:
                rtstruct_folder = os.path.join(root, 'RTSTRUCT')
                rtstruct_files = glob(os.path.join(rtstruct_folder, '*'))
                if rtstruct_files:
                    rtstruct_file = rtstruct_files[0]
        
        if ct_folder and rtstruct_file:
            try:
                result = analyze_rtstruct_ct_match(ct_folder, rtstruct_file)
                if result:
                    result['patient_id'] = patient_id
                    results_list.append(result)
            except Exception as e:
                print(f"Error processing {patient_id}: {str(e)}")
    
    if results_list and output_csv:
        df = pd.DataFrame(results_list)
        df.to_csv(output_csv, index=False, sep=';')
        print(f"\nResults saved to: {output_csv}")
    
    return results_list


# Example usage
if __name__ == "__main__":

    # Single patient analysis
    #contrasct
    ct_folder = r"\\zkh\appdata\RTDicom\Projectline_HNC_modelling\PRI2MA\Now_Has_endpoint_for_at_least_1_toxicity\9980149\with_contrast\CT"
    # no contrast
    #ct_folder = r"\\zkh\appdata\RTDicom\Projectline_HNC_modelling\PRI2MA\Now_Has_endpoint_for_at_least_1_toxicity\9980149\no_contrast\CT"
    
    rtstruct_path = r"\\zkh\appdata\RTDicom\Projectline_HNC_modelling\PRI2MA\Now_Has_endpoint_for_at_least_1_toxicity\9980149\no_contrast\RTSTRUCT\IM1 (1).DCM"
    
    # Uncomment to run:
    results = analyze_rtstruct_ct_match(ct_folder, rtstruct_path)

    print(results)
    
    # Batch analysis
    # base_dir = "path/to/all/patients"
    # check_all_patients(base_dir, output_csv="rtstruct_ct_matching_report.csv")