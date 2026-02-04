import os
import pydicom


def check_frame_of_reference_uid(ct_folder, rtdose_file, rtstruct_file):
    """
    Check if CT, RTDOSE, and RTSTRUCT have matching Frame of Reference UID.
    This is crucial for proper registration.
    
    Args:
        ct_folder: Path to CT DICOM folder
        rtdose_file: Path to RTDOSE file
        rtstruct_file: Path to RTSTRUCT file
        
    Returns:
        bool: True if all UIDs match
    """
    print("\nChecking Frame of Reference UIDs...")
    
    # Get CT UID (from first slice)
    ct_files = [f for f in os.listdir(ct_folder) if f.endswith('.dcm') or not '.' in f]
    if ct_files:
        ct_ds = pydicom.dcmread(os.path.join(ct_folder, ct_files[0]))
        ct_uid = ct_ds.FrameOfReferenceUID if hasattr(ct_ds, 'FrameOfReferenceUID') else None
    else:
        ct_uid = None
    
    # Get RTDOSE UID
    dose_ds = pydicom.dcmread(rtdose_file)
    dose_uid = dose_ds.FrameOfReferenceUID if hasattr(dose_ds, 'FrameOfReferenceUID') else None
    
    # Get RTSTRUCT UID
    struct_ds = pydicom.dcmread(rtstruct_file)
    struct_uid = struct_ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID if hasattr(struct_ds, 'ReferencedFrameOfReferenceSequence') else None
    
    print(f"CT Frame of Reference UID: {ct_uid}")
    print(f"RTDOSE Frame of Reference UID: {dose_uid}")
    print(f"RTSTRUCT Frame of Reference UID: {struct_uid}")
    
    match = (ct_uid == dose_uid == struct_uid)
    print(f"UIDs match: {match}")
    
    return match