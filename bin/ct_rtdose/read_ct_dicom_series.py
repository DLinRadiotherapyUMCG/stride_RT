import SimpleITK as sitk



def read_ct_dicom_series(ct_folder_path):
    """
    Read CT DICOM series from a folder.
    
    Args:
        ct_folder_path: Path to folder containing CT DICOM slices
        
    Returns:
        ct_array: 3D numpy array with shape (num_slices, height, width)
        ct_image: SimpleITK image object with metadata (spacing, origin, direction)
    """
    print(f"Reading CT DICOM series from: {ct_folder_path}")
    
    # Get DICOM file names
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(ct_folder_path)
    
    if not dicom_names:
        raise ValueError(f"No DICOM files found in {ct_folder_path}")
    
    print(f"Found {len(dicom_names)} CT slices")
    
    # Read the series
    reader.SetFileNames(dicom_names)
    ct_image = reader.Execute()
    
    print(f"CT array shape: {ct_image.GetSize()}")
    print(f"CT spacing: {ct_image.GetSpacing()}")  # (x, y, z)
    print(f"CT origin: {ct_image.GetOrigin()}")
    
    return ct_image