import numpy as np
import SimpleITK as sitk

def convert_image_to_numpy(image):
    # helper function, extracts a numpy array from an SimpleITK image
    return sitk.GetArrayFromImage(image)





def convert_segmentation_map_to_image(segmentation_map, reference_image):
    """
    Convert segmentation map numpy array to SimpleITK image with reference metadata.
    
    Args:
        segmentation_map: 3D numpy array with integer labels
        reference_image: SimpleITK image to copy metadata from (i.e. CT image)
        
    Returns:
        segmentation_image: SimpleITK image with segmentation data
    """
    segmentation_image = sitk.GetImageFromArray(segmentation_map.astype(np.uint16))
    segmentation_image.SetSpacing(reference_image.GetSpacing())
    segmentation_image.SetOrigin(reference_image.GetOrigin())
    segmentation_image.SetDirection(reference_image.GetDirection())
    
    return segmentation_image




import SimpleITK as sitk
import numpy as np
from pydicom import dcmread
from typing import List, Tuple

def pydicom_ct_to_sitk(ct_datasets: List, apply_modality_lut: bool = True) -> sitk.Image:
    """
    Convert already-loaded pydicom CT datasets to SimpleITK Image.
    
    Args:
        ct_datasets: List of pydicom Dataset objects (already loaded with dcmread)
        apply_modality_lut: Whether to apply HU conversion
        
    Returns:
        SimpleITK Image
    """
    # Sort by z-position
    ct_datasets = sorted(ct_datasets, key=lambda x: float(x.ImagePositionPatient[2]))
    
    # Stack pixel arrays
    pixel_array = np.stack([ds.pixel_array for ds in ct_datasets], axis=0)
    
    # Apply modality LUT if requested
    if apply_modality_lut:
        slope = float(ct_datasets[0].RescaleSlope) if hasattr(ct_datasets[0], 'RescaleSlope') else 1.0
        intercept = float(ct_datasets[0].RescaleIntercept) if hasattr(ct_datasets[0], 'RescaleIntercept') else -1024.0
        pixel_array = pixel_array * slope + intercept
        pixel_array = pixel_array.astype(np.int16)
    
    # Create SimpleITK image from numpy array
    sitk_image = sitk.GetImageFromArray(pixel_array)
    
    # Set geometry from DICOM metadata
    first_slice = ct_datasets[0]
    
    # Spacing
    pixel_spacing = first_slice.PixelSpacing
    z_positions = [float(ds.ImagePositionPatient[2]) for ds in ct_datasets]
    z_spacing = abs(z_positions[-1] - z_positions[0]) / (len(z_positions) - 1) if len(z_positions) > 1 else 1.0
    sitk_image.SetSpacing([float(pixel_spacing[0]), float(pixel_spacing[1]), z_spacing])
    
    # Origin
    origin = [float(x) for x in first_slice.ImagePositionPatient]
    sitk_image.SetOrigin(origin)
    
    # Direction
    if hasattr(first_slice, 'ImageOrientationPatient'):
        iop = first_slice.ImageOrientationPatient
        # Create 3x3 direction matrix
        row_cosines = np.array([float(iop[0]), float(iop[1]), float(iop[2])])
        col_cosines = np.array([float(iop[3]), float(iop[4]), float(iop[5])])
        
        # Cross product for slice direction
        slice_cosines = np.cross(row_cosines, col_cosines)
        
        # Flatten to 1D for SimpleITK (column-major order)
        direction = np.array([
            row_cosines[0], col_cosines[0], slice_cosines[0],
            row_cosines[1], col_cosines[1], slice_cosines[1],
            row_cosines[2], col_cosines[2], slice_cosines[2]
        ])
        sitk_image.SetDirection(direction.tolist())
    
    return sitk_image


def pydicom_rtdose_to_sitk(rtdose_dataset) -> sitk.Image:
    """
    Convert already-loaded pydicom RTDOSE dataset to SimpleITK Image.
    
    Args:
        rtdose_dataset: pydicom Dataset object (already loaded with dcmread)
        
    Returns:
        SimpleITK Image with raw dose values (not converted to cGy yet)
    """
    # Get pixel array
    pixel_array = rtdose_dataset.pixel_array
    
    # Handle 4D arrays (multi-frame)
    if pixel_array.ndim == 4:
        pixel_array = pixel_array[..., -1]
    
    # Create SimpleITK image
    sitk_image = sitk.GetImageFromArray(pixel_array)
    
    # Set geometry
    pixel_spacing = rtdose_dataset.PixelSpacing
    grid_frame_offset = rtdose_dataset.GridFrameOffsetVector if hasattr(rtdose_dataset, 'GridFrameOffsetVector') else []
    
    if len(grid_frame_offset) > 1:
        z_spacing = abs(float(grid_frame_offset[1]) - float(grid_frame_offset[0]))
    else:
        z_spacing = 1.0
    
    sitk_image.SetSpacing([float(pixel_spacing[0]), float(pixel_spacing[1]), z_spacing])
    
    # Origin
    if hasattr(rtdose_dataset, 'ImagePositionPatient'):
        origin = [float(x) for x in rtdose_dataset.ImagePositionPatient]
        sitk_image.SetOrigin(origin)
    
    # Direction
    if hasattr(rtdose_dataset, 'ImageOrientationPatient'):
        iop = rtdose_dataset.ImageOrientationPatient
        row_cosines = np.array([float(iop[0]), float(iop[1]), float(iop[2])])
        col_cosines = np.array([float(iop[3]), float(iop[4]), float(iop[5])])
        slice_cosines = np.cross(row_cosines, col_cosines)
        
        direction = np.array([
            row_cosines[0], col_cosines[0], slice_cosines[0],
            row_cosines[1], col_cosines[1], slice_cosines[1],
            row_cosines[2], col_cosines[2], slice_cosines[2]
        ])
        sitk_image.SetDirection(direction.tolist())
    
    return sitk_image
