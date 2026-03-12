import pydicom
import SimpleITK as sitk

from src.utils.resample_to_reference import resample_to_reference

def read_rtdose_dicom(rtdose_file_path, ct_image=None):
    """
    Read RTDOSE DICOM file.
    
    Args:
        rtdose_file_path: Path to RTDOSE DICOM file
        ct_image: Optional CT image for registration/resampling
        
    Returns:
        dose_array: 3D numpy array with dose values in cGy
        dose_image: SimpleITK image object
    """
    print(f"\nReading RTDOSE from: {rtdose_file_path}")
    
    # Read RTDOSE using SimpleITK
    dose_image = sitk.ReadImage(rtdose_file_path)
    
    # Read with pydicom to get dose grid scaling
    ds = pydicom.dcmread(rtdose_file_path)
    dose_grid_scaling = float(ds.DoseGridScaling)
    
    print(f"RTDOSE array shape: {dose_image.GetSize()}")
    dose_arrary = sitk.GetArrayFromImage(dose_image)
    dose_array = dose_arrary * dose_grid_scaling * 100  # to cGy
    print(f"RTDOSE min/max: {dose_array.min():.2f} / {dose_array.max():.2f} cGy")
    print(f"RTDOSE spacing: {dose_image.GetSpacing()}")
    
    # Optionally resample RTDOSE to CT grid
    if ct_image is not None:
        print("Resampling RTDOSE to CT grid...")
        dose_image_resampled = resample_to_reference(dose_image, ct_image)
        
        print(f"Resampled RTDOSE shape: {dose_image_resampled.GetSize()}")
        return dose_image_resampled
    
    return dose_array, dose_image