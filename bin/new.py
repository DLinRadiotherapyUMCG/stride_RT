"""
DICOM Preprocessing Module with RTSTRUCT Support - Extended Version

This module handles preprocessing of CT, RTDOSE, and RTSTRUCT DICOM files for 
radiotherapy treatment planning. It loads CT scans, radiation dose distributions,
and structure sets (organ contours), resamples them to a common grid, and saves
the processed arrays along with metadata.

New Features in this version:
- RTSTRUCT loading and processing
- Structure mask generation from contours
- Multi-structure segmentation maps
- Bounding box calculation for cropping
- Support for merging multiple RTSTRUCT files

Author: Extended version with RTSTRUCT support
Date: 2026
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Set
from dataclasses import dataclass, asdict, field
from multiprocessing import Pool
import warnings

import numpy as np
import pandas as pd
import SimpleITK as sitk
from pydicom import dcmread
from tqdm import tqdm
from rt_utils import RTStructBuilder
import skimage.draw


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CTMetadata:
    """Structured metadata for CT images."""
    patient_id: str
    path: str
    direction: Tuple[float, ...]
    origin: Tuple[float, ...]
    size: Tuple[int, ...]
    spacing: Tuple[float, ...]
    slope: float
    intercept: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class RTDoseMetadata:
    """Structured metadata for RTDOSE images."""
    patient_id: str
    path_file: str
    path_dir: str
    dose_grid_scaling: float
    dose_units: str
    direction: Tuple[float, ...]
    origin: Tuple[float, ...]
    size: Tuple[int, ...]
    spacing: Tuple[float, ...]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class RTStructMetadata:
    """Structured metadata for RTSTRUCT images."""
    patient_id: str
    path_files: List[str]
    structure_names: List[str]
    structure_colors: Dict[str, Tuple[int, int, int]]
    structure_volumes: Dict[str, float]
    structure_bounding_boxes: Dict[str, Dict[str, int]]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class StructureSet:
    """Container for structure set information."""
    name: str
    mask: np.ndarray
    color: Tuple[int, int, int]
    roi_number: int
    volume_mm3: float = 0.0
    bounding_box: Dict[str, int] = field(default_factory=dict)


class DICOMPreprocessor:
    """
    Main class for DICOM preprocessing operations including RTSTRUCT support.
    
    Handles loading, resampling, and saving of CT, RTDOSE, and RTSTRUCT DICOM files.
    Supports single-patient and batch processing with optional multiprocessing.
    """
    
    def __init__(
        self,
        data_dir: Path,
        save_dir: Path,
        ct_dtype: np.dtype = np.int16,
        rtdose_dtype: np.dtype = np.float32,
        segmentation_dtype: np.dtype = np.uint8,
        interpolator: int = sitk.sitkLinear,
        num_workers: int = 1,
        structure_name_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Initialize DICOM preprocessor with RTSTRUCT support.
        
        Args:
            data_dir: Root directory containing patient DICOM folders
            save_dir: Directory to save processed arrays and metadata
            ct_dtype: Data type for CT arrays (default: int16 for HU values)
            rtdose_dtype: Data type for RTDOSE arrays (default: float32 for cGy)
            segmentation_dtype: Data type for segmentation masks (default: uint8)
            interpolator: SimpleITK interpolator for resampling (default: linear)
            num_workers: Number of parallel workers (1 = single process)
            structure_name_mapping: Optional dict to standardize structure names
        """
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.ct_dtype = ct_dtype
        self.rtdose_dtype = rtdose_dtype
        self.segmentation_dtype = segmentation_dtype
        self.interpolator = interpolator
        self.num_workers = num_workers
        self.structure_name_mapping = structure_name_mapping or {}
        
        # Create output directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized preprocessor: {self.data_dir} -> {self.save_dir}")
    
    def standardize_structure_name(self, name: str) -> str:
        """
        Standardize structure names using mapping.
        
        Args:
            name: Original structure name
            
        Returns:
            Standardized structure name
        """
        name_lower = name.lower().strip()
        
        # Check if we have a mapping for this name
        for pattern, standard_name in self.structure_name_mapping.items():
            if pattern.lower() in name_lower:
                return standard_name
        
        # Return cleaned version if no mapping found
        return name.strip()
    
    def load_ct_image(self, ct_path: Path) -> Tuple[sitk.Image, Dict]:
        """
        Load CT DICOM series as SimpleITK Image with proper HU conversion.
        
        This function:
        1. Reads all DICOM slices in the CT directory
        2. Computes correct z-spacing from slice positions (not header)
        3. Applies modality LUT to convert pixel values to Hounsfield Units
        
        Args:
            ct_path: Path to directory containing CT DICOM files
            
        Returns:
            Tuple of (SimpleITK Image in HU, metadata dict)
        """
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(str(ct_path))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {ct_path}")
        
        reader.SetFileNames(dicom_files)
        image = reader.Execute()
        
        # Load first slice for metadata extraction
        first_slice = dcmread(dicom_files[0])
        metadata = json.loads(first_slice.to_json())
        
        # Extract pixel spacing (x, y) from DICOM header
        pixel_spacing = metadata.get('00280030', {}).get('Value', [1.0, 1.0])
        xy_spacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
        
        # Compute z-spacing from actual slice positions
        z_coords = []
        for dcm_file in dicom_files:
            ds = dcmread(dcm_file)
            position = ds.ImagePositionPatient
            z_coords.append(float(position[2]))
        
        z_coords_sorted = sorted(z_coords)
        if len(z_coords_sorted) > 1:
            z_spacing = abs(z_coords_sorted[-1] - z_coords_sorted[0]) / (len(z_coords_sorted) - 1)
        else:
            z_spacing = float(metadata.get('00180050', {}).get('Value', [1.0])[0])
        
        if not (0.5 < z_spacing < 5.0):
            warnings.warn(f"Unusual z-spacing detected: {z_spacing} mm")
        
        spacing = xy_spacing + [z_spacing]
        image.SetSpacing(spacing)
        
        # Extract modality LUT parameters for HU conversion
        slope = float(metadata.get('00281053', {}).get('Value', [1.0])[0])
        intercept = float(metadata.get('00281052', {}).get('Value', [-1024.0])[0])
        
        # Convert to numpy and apply modality LUT
        arr = sitk.GetArrayFromImage(image)
        arr_hu = arr * slope + intercept
        arr_hu = arr_hu.astype(self.ct_dtype)
        
        # Create new SimpleITK image with HU values
        image_hu = sitk.GetImageFromArray(arr_hu)
        image_hu.SetOrigin(image.GetOrigin())
        image_hu.SetDirection(image.GetDirection())
        image_hu.SetSpacing(spacing)
        
        return image_hu, metadata
    
    def load_rtdose_image(self, rtdose_path: Path) -> sitk.Image:
        """
        Load RTDOSE DICOM file as SimpleITK Image.
        
        Args:
            rtdose_path: Path to directory containing RTDOSE DICOM file
            
        Returns:
            SimpleITK Image
        """
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(str(rtdose_path))
        
        if not dicom_files:
            raise ValueError(f"No RTDOSE files found in {rtdose_path}")
        
        reader.SetFileNames(dicom_files)
        image = reader.Execute()
        
        # Extract last frame if multi-frame RTDOSE
        if image.GetDepth() > 1 and len(image.GetSize()) > 3:
            if image.GetSize()[3] > 1:
                image = image[:, :, :, -1]
        
        return image
    
    def load_rtstruct(
        self, 
        rtstruct_paths: List[Path], 
        ct_path: Path
    ) -> Tuple[List[StructureSet], sitk.Image]:
        """
        Load RTSTRUCT file(s) and generate 3D masks.
        
        This function can merge multiple RTSTRUCT files (e.g., auto-segmentation
        and manual corrections).
        
        Args:
            rtstruct_paths: List of paths to RTSTRUCT DICOM files
            ct_path: Path to CT directory (needed for reference geometry)
            
        Returns:
            Tuple of (list of StructureSet objects, reference CT image)
        """
        # Load reference CT image for geometry
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(str(ct_path))
        reader.SetFileNames(dicom_files)
        ct_image = reader.Execute()
        
        ct_size = ct_image.GetSize()  # (x, y, z)
        ct_spacing = ct_image.GetSpacing()
        ct_origin = ct_image.GetOrigin()
        ct_direction = ct_image.GetDirection()
        
        # Dictionary to store structures (allows merging from multiple files)
        all_structures = {}
        
        for rtstruct_path in rtstruct_paths:
            try:
                # Load RTSTRUCT using rt-utils
                rtstruct = RTStructBuilder.create_from(
                    dicom_series_path=str(ct_path),
                    rt_struct_path=str(rtstruct_path)
                )
                
                # Get all ROI names
                roi_names = rtstruct.get_roi_names()
                
                for roi_name in roi_names:
                    # Standardize name
                    std_name = self.standardize_structure_name(roi_name)
                    
                    # Get 3D mask for this structure
                    mask_3d = rtstruct.get_roi_mask_by_name(roi_name)
                    
                    # Get color
                    ds = dcmread(rtstruct_path)
                    color = (128, 128, 128)  # Default gray
                    for roi_seq in ds.StructureSetROISequence:
                        if roi_seq.ROIName == roi_name:
                            roi_number = roi_seq.ROINumber
                            for contour_seq in ds.ROIContourSequence:
                                if contour_seq.ReferencedROINumber == roi_number:
                                    if hasattr(contour_seq, 'ROIDisplayColor'):
                                        color = tuple(contour_seq.ROIDisplayColor)
                                    break
                            break
                    
                    # If structure already exists, merge masks (logical OR)
                    if std_name in all_structures:
                        all_structures[std_name].mask = np.logical_or(
                            all_structures[std_name].mask,
                            mask_3d
                        ).astype(np.uint8)
                    else:
                        all_structures[std_name] = StructureSet(
                            name=std_name,
                            mask=mask_3d.astype(np.uint8),
                            color=color,
                            roi_number=roi_number
                        )
                
                logger.info(f"Loaded {len(roi_names)} structures from {rtstruct_path.name}")
                
            except Exception as e:
                logger.warning(f"Failed to load RTSTRUCT {rtstruct_path}: {e}")
                continue
        
        # Calculate volumes and bounding boxes for each structure
        structure_list = []
        for struct in all_structures.values():
            # Calculate volume
            voxel_volume_mm3 = np.prod(ct_spacing)
            struct.volume_mm3 = float(np.sum(struct.mask) * voxel_volume_mm3)
            
            # Calculate bounding box
            coords = np.where(struct.mask > 0)
            if len(coords[0]) > 0:
                struct.bounding_box = {
                    'z_min': int(np.min(coords[0])),
                    'z_max': int(np.max(coords[0])),
                    'y_min': int(np.min(coords[1])),
                    'y_max': int(np.max(coords[1])),
                    'x_min': int(np.min(coords[2])),
                    'x_max': int(np.max(coords[2]))
                }
            else:
                struct.bounding_box = {
                    'z_min': 0, 'z_max': 0,
                    'y_min': 0, 'y_max': 0,
                    'x_min': 0, 'x_max': 0
                }
            
            structure_list.append(struct)
        
        return structure_list, ct_image
    
    def create_multi_label_segmentation(
        self,
        structures: List[StructureSet],
        structure_priority: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Create a single multi-label segmentation map from multiple structures.
        
        Args:
            structures: List of StructureSet objects
            structure_priority: Optional list defining structure priority for overlaps.
                               Structures earlier in list take precedence.
                               
        Returns:
            3D numpy array where each voxel contains the structure label (0 = background)
        """
        if not structures:
            raise ValueError("No structures provided")
        
        # Get shape from first structure
        shape = structures[0].mask.shape
        segmentation = np.zeros(shape, dtype=self.segmentation_dtype)
        
        # Create mapping from structure name to label
        if structure_priority:
            # Use provided priority order
            structure_order = []
            for name in reversed(structure_priority):  # Reverse so high priority is last
                matching = [s for s in structures if s.name.lower() == name.lower()]
                if matching:
                    structure_order.extend(matching)
            # Add any structures not in priority list
            for s in structures:
                if s not in structure_order:
                    structure_order.append(s)
        else:
            # Default: sort by volume (smaller structures last, so they take precedence)
            structure_order = sorted(structures, key=lambda s: s.volume_mm3, reverse=True)
        
        # Apply masks in order (later structures overwrite earlier ones)
        for idx, struct in enumerate(structure_order, start=1):
            segmentation[struct.mask > 0] = idx
        
        return segmentation
    
    def calculate_combined_bounding_box(
        self,
        structures: List[StructureSet],
        margin_mm: Tuple[float, float, float] = (10.0, 10.0, 10.0),
        ct_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        ct_size: Tuple[int, int, int] = None
    ) -> Dict[str, int]:
        """
        Calculate bounding box that encompasses all specified structures with margin.
        
        Args:
            structures: List of structures to include
            margin_mm: Margin in mm to add in each direction (z, y, x)
            ct_spacing: CT spacing in mm (z, y, x)
            ct_size: CT size in voxels (z, y, x) for clamping
            
        Returns:
            Dictionary with keys: z_min, z_max, y_min, y_max, x_min, x_max
        """
        if not structures:
            raise ValueError("No structures provided for bounding box calculation")
        
        # Find overall min/max across all structures
        z_min = min(s.bounding_box['z_min'] for s in structures)
        z_max = max(s.bounding_box['z_max'] for s in structures)
        y_min = min(s.bounding_box['y_min'] for s in structures)
        y_max = max(s.bounding_box['y_max'] for s in structures)
        x_min = min(s.bounding_box['x_min'] for s in structures)
        x_max = max(s.bounding_box['x_max'] for s in structures)
        
        # Convert margin from mm to voxels
        margin_voxels = (
            int(margin_mm[0] / ct_spacing[2]),  # z
            int(margin_mm[1] / ct_spacing[1]),  # y
            int(margin_mm[2] / ct_spacing[0])   # x
        )
        
        # Add margin
        z_min = max(0, z_min - margin_voxels[0])
        z_max = z_max + margin_voxels[0]
        y_min = max(0, y_min - margin_voxels[1])
        y_max = y_max + margin_voxels[1]
        x_min = max(0, x_min - margin_voxels[2])
        x_max = x_max + margin_voxels[2]
        
        # Clamp to CT size if provided
        if ct_size:
            z_max = min(z_max, ct_size[0] - 1)
            y_max = min(y_max, ct_size[1] - 1)
            x_max = min(x_max, ct_size[2] - 1)
        
        return {
            'z_min': z_min,
            'z_max': z_max,
            'y_min': y_min,
            'y_max': y_max,
            'x_min': x_min,
            'x_max': x_max
        }
    
    def crop_to_bounding_box(
        self,
        array: np.ndarray,
        bounding_box: Dict[str, int]
    ) -> np.ndarray:
        """
        Crop array to bounding box.
        
        Args:
            array: 3D numpy array (z, y, x)
            bounding_box: Dict with z_min, z_max, y_min, y_max, x_min, x_max
            
        Returns:
            Cropped array
        """
        return array[
            bounding_box['z_min']:bounding_box['z_max']+1,
            bounding_box['y_min']:bounding_box['y_max']+1,
            bounding_box['x_min']:bounding_box['x_max']+1
        ]
    
    def resample_rtdose_to_ct_grid(
        self,
        rtdose_image: sitk.Image,
        ct_image: sitk.Image
    ) -> Tuple[sitk.Image, np.ndarray, np.ndarray, np.ndarray]:
        """
        Resample RTDOSE to match CT grid (spacing, size, origin, direction).
        
        Args:
            rtdose_image: RTDOSE SimpleITK Image
            ct_image: CT SimpleITK Image (reference grid)
            
        Returns:
            Tuple of (resampled image, resampled array, CT array, original RTDOSE array)
        """
        resampled_image = sitk.Resample(
            rtdose_image,
            ct_image,
            transform=sitk.Transform(),
            interpolator=self.interpolator,
            defaultPixelValue=0,
            outputPixelType=rtdose_image.GetPixelID()
        )
        
        arr_rtdose = sitk.GetArrayFromImage(rtdose_image)
        arr_resampled = sitk.GetArrayFromImage(resampled_image)
        arr_ct = sitk.GetArrayFromImage(ct_image)
        
        assert arr_ct.shape == arr_resampled.shape
        
        return resampled_image, arr_resampled, arr_ct, arr_rtdose
    
    def process_patient(
        self,
        patient_id: str,
        patient_folder: Path,
        process_rtstruct: bool = True,
        crop_to_structures: Optional[List[str]] = None,
        crop_margin_mm: Tuple[float, float, float] = (10.0, 10.0, 10.0),
        verbose: bool = True
    ) -> bool:
        """
        Process single patient: load CT, RTDOSE, RTSTRUCT, resample, and save.
        
        Args:
            patient_id: Patient identifier
            patient_folder: Path to patient's DICOM folder
            process_rtstruct: Whether to process RTSTRUCT files
            crop_to_structures: Optional list of structure names to crop around
            crop_margin_mm: Margin in mm when cropping (z, y, x)
            verbose: Whether to print progress messages
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if verbose:
                logger.info(f"Processing patient {patient_id}")
            
            # Find CT, RTDOSE, and RTSTRUCT directories
            ct_path = None
            rtdose_path = None
            rtstruct_paths = []
            
            for subdir in patient_folder.rglob('*'):
                if subdir.is_dir():
                    dir_name = subdir.name.upper()
                    if dir_name == 'CT':
                        ct_path = subdir
                    elif dir_name == 'RTDOSE':
                        rtdose_path = subdir
                    elif dir_name == 'RTSTRUCT':
                        # Find all RTSTRUCT files in this directory
                        rtstruct_files = list(subdir.glob('*.dcm')) or list(subdir.glob('*'))
                        rtstruct_paths.extend([f for f in rtstruct_files if f.is_file()])
            
            if ct_path is None:
                raise FileNotFoundError(f"CT directory not found for patient {patient_id}")
            if rtdose_path is None:
                raise FileNotFoundError(f"RTDOSE directory not found for patient {patient_id}")
            
            # Load CT
            if verbose:
                logger.info(f"  Loading CT from {ct_path}")
            ct_image, ct_metadata_raw = self.load_ct_image(ct_path)
            
            # Load RTDOSE
            if verbose:
                logger.info(f"  Loading RTDOSE from {rtdose_path}")
            rtdose_image = self.load_rtdose_image(rtdose_path)
            
            # Get RTDOSE metadata
            rtdose_files = list(rtdose_path.glob('*'))
            if not rtdose_files:
                raise FileNotFoundError(f"No RTDOSE files in {rtdose_path}")
            rtdose_file = rtdose_files[0]
            rtdose_ds = dcmread(rtdose_file)
            dose_grid_scaling = float(rtdose_ds.DoseGridScaling)
            dose_units = str(rtdose_ds.DoseUnits)
            
            # Resample RTDOSE to CT grid
            if verbose:
                logger.info(f"  Resampling RTDOSE to CT grid")
            resampled_image, arr_rtdose_resampled, arr_ct, arr_rtdose_orig = \
                self.resample_rtdose_to_ct_grid(rtdose_image, ct_image)
            
            # Convert RTDOSE to cGy
            arr_rtdose_resampled_cgy = (arr_rtdose_resampled * dose_grid_scaling * 100).astype(self.rtdose_dtype)
            
            # Process RTSTRUCT if requested and available
            structures = []
            segmentation_map = None
            rtstruct_meta = None
            bounding_box = None
            
            if process_rtstruct and rtstruct_paths:
                if verbose:
                    logger.info(f"  Loading {len(rtstruct_paths)} RTSTRUCT file(s)")
                
                try:
                    structures, _ = self.load_rtstruct(rtstruct_paths, ct_path)
                    
                    if structures:
                        # Create multi-label segmentation map
                        segmentation_map = self.create_multi_label_segmentation(structures)
                        
                        # Create RTSTRUCT metadata
                        rtstruct_meta = RTStructMetadata(
                            patient_id=patient_id,
                            path_files=[str(p) for p in rtstruct_paths],
                            structure_names=[s.name for s in structures],
                            structure_colors={s.name: s.color for s in structures},
                            structure_volumes={s.name: s.volume_mm3 for s in structures},
                            structure_bounding_boxes={s.name: s.bounding_box for s in structures}
                        )
                        
                        if verbose:
                            logger.info(f"  Created segmentation map with {len(structures)} structures")
                        
                        # Calculate cropping bounding box if requested
                        if crop_to_structures:
                            crop_structures = [s for s in structures 
                                             if any(cs.lower() in s.name.lower() 
                                                   for cs in crop_to_structures)]
                            
                            if crop_structures:
                                ct_spacing = ct_image.GetSpacing()
                                ct_size_sitk = ct_image.GetSize()
                                ct_size_np = (ct_size_sitk[2], ct_size_sitk[1], ct_size_sitk[0])  # z,y,x
                                
                                bounding_box = self.calculate_combined_bounding_box(
                                    crop_structures,
                                    margin_mm=crop_margin_mm,
                                    ct_spacing=(ct_spacing[2], ct_spacing[1], ct_spacing[0]),  # z,y,x
                                    ct_size=ct_size_np
                                )
                                
                                if verbose:
                                    logger.info(f"  Calculated bounding box for cropping: {bounding_box}")
                                
                                # Crop all arrays
                                arr_ct = self.crop_to_bounding_box(arr_ct, bounding_box)
                                arr_rtdose_resampled_cgy = self.crop_to_bounding_box(
                                    arr_rtdose_resampled_cgy, bounding_box
                                )
                                segmentation_map = self.crop_to_bounding_box(
                                    segmentation_map, bounding_box
                                )
                                
                                if verbose:
                                    logger.info(f"  Cropped to shape: {arr_ct.shape}")
                            else:
                                logger.warning(f"  No structures found matching crop list: {crop_to_structures}")
                
                except Exception as e:
                    logger.warning(f"  Failed to process RTSTRUCT: {e}")
            
            # Create metadata objects
            ct_meta = CTMetadata(
                patient_id=patient_id,
                path=str(ct_path),
                direction=ct_image.GetDirection(),
                origin=ct_image.GetOrigin(),
                size=ct_image.GetSize(),
                spacing=ct_image.GetSpacing(),
                slope=float(ct_metadata_raw.get('00281053', {}).get('Value', [1.0])[0]),
                intercept=float(ct_metadata_raw.get('00281052', {}).get('Value', [-1024.0])[0])
            )
            
            rtdose_meta = RTDoseMetadata(
                patient_id=patient_id,
                path_file=str(rtdose_file),
                path_dir=str(rtdose_path),
                dose_grid_scaling=dose_grid_scaling,
                dose_units=dose_units,
                direction=resampled_image.GetDirection(),
                origin=resampled_image.GetOrigin(),
                size=resampled_image.GetSize(),
                spacing=resampled_image.GetSpacing()
            )
            
            # Save outputs
            patient_save_dir = self.save_dir / patient_id
            patient_save_dir.mkdir(parents=True, exist_ok=True)
            
            if verbose:
                logger.info(f"  Saving to {patient_save_dir}")
            
            # Save arrays
            np.save(patient_save_dir / 'ct.npy', arr_ct)
            np.save(patient_save_dir / 'rtdose.npy', arr_rtdose_resampled_cgy)
            
            if segmentation_map is not None:
                np.save(patient_save_dir / 'segmentation.npy', segmentation_map)
            
            # Save metadata
            with open(patient_save_dir / 'ct_metadata.json', 'w') as f:
                json.dump(ct_meta.to_dict(), f, indent=2, default=str)
            
            with open(patient_save_dir / 'rtdose_metadata.json', 'w') as f:
                json.dump(rtdose_meta.to_dict(), f, indent=2, default=str)
            
            if rtstruct_meta is not None:
                with open(patient_save_dir / 'rtstruct_metadata.json', 'w') as f:
                    json.dump(rtstruct_meta.to_dict(), f, indent=2, default=str)
            
            if bounding_box is not None:
                with open(patient_save_dir / 'bounding_box.json', 'w') as f:
                    json.dump(bounding_box, f, indent=2)
            
            if verbose:
                logger.info(f"  Successfully processed patient {patient_id}")
                logger.info(f"    CT shape: {arr_ct.shape}")
                logger.info(f"    RTDOSE shape: {arr_rtdose_resampled_cgy.shape}")
                if segmentation_map is not None:
                    logger.info(f"    Segmentation shape: {segmentation_map.shape}")
                    logger.info(f"    Number of structures: {len(structures)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process patient {patient_id}: {str(e)}")
            import traceback
            if verbose:
                logger.error(traceback.format_exc())
            return False
    
    def process_all_patients(
        self,
        patient_list: Optional[List[str]] = None,
        process_rtstruct: bool = True,
        crop_to_structures: Optional[List[str]] = None,
        crop_margin_mm: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    ) -> pd.DataFrame:
        """
        Process all patients in the data directory.
        
        Args:
            patient_list: Optional list of specific patient IDs to process
            process_rtstruct: Whether to process RTSTRUCT files
            crop_to_structures: Optional list of structure names to crop around
            crop_margin_mm: Margin in mm when cropping (z, y, x)
            
        Returns:
            DataFrame with processing results
        """
        if patient_list is None:
            patient_folders = [p for p in self.data_dir.iterdir() if p.is_dir()]
            patient_ids = [p.name for p in patient_folders]
        else:
            patient_ids = patient_list
            patient_folders = [self.data_dir / pid for pid in patient_ids]
        
        logger.info(f"Found {len(patient_ids)} patients to process")
        
        if self.num_workers == 1:
            results = []
            for patient_id, patient_folder in tqdm(
                zip(patient_ids, patient_folders),
                total=len(patient_ids),
                desc="Processing patients"
            ):
                success = self.process_patient(
                    patient_id, patient_folder, 
                    process_rtstruct=process_rtstruct,
                    crop_to_structures=crop_to_structures,
                    crop_margin_mm=crop_margin_mm,
                    verbose=False
                )
                results.append({'patient_id': patient_id, 'success': success})
        else:
            logger.info(f"Using {self.num_workers} parallel workers")
            with Pool(processes=self.num_workers) as pool:
                args = [
                    (pid, pf, process_rtstruct, crop_to_structures, crop_margin_mm, False) 
                    for pid, pf in zip(patient_ids, patient_folders)
                ]
                results_bool = list(tqdm(
                    pool.starmap(self.process_patient, args),
                    total=len(args),
                    desc="Processing patients"
                ))
                results = [
                    {'patient_id': pid, 'success': success}
                    for pid, success in zip(patient_ids, results_bool)
                ]
        
        df_results = pd.DataFrame(results)
        
        n_success = df_results['success'].sum()
        n_failed = len(df_results) - n_success
        logger.info(f"\nProcessing complete:")
        logger.info(f"  Successful: {n_success}/{len(df_results)}")
        logger.info(f"  Failed: {n_failed}/{len(df_results)}")
        
        if n_failed > 0:
            failed_patients = df_results[~df_results['success']]['patient_id'].tolist()
            logger.warning(f"  Failed patients: {failed_patients}")
        
        return df_results


def main():
    """
    Example usage of the DICOM preprocessor with RTSTRUCT support.
    """
    
    # Configuration
    DATA_DIR = Path("/path/to/dicom/data")
    SAVE_DIR = Path("/path/to/output")
    NUM_WORKERS = 4
    
    # Define structure name mapping for standardization
    structure_mapping = {
        'parotid_l': 'Parotid_L',
        'parotid_r': 'Parotid_R',
        'parotid left': 'Parotid_L',
        'parotid right': 'Parotid_R',
        'submandibular_l': 'Submandibular_L',
        'submandibular_r': 'Submandibular_R',
        'spinal': 'SpinalCord',
        'cord': 'SpinalCord',
        'brain': 'BrainStem',
        'brainstem': 'BrainStem'
    }
    
    # Create preprocessor with RTSTRUCT support
    preprocessor = DICOMPreprocessor(
        data_dir=DATA_DIR,
        save_dir=SAVE_DIR,
        num_workers=NUM_WORKERS,
        structure_name_mapping=structure_mapping
    )
    
    # Example 1: Process all patients with RTSTRUCT
    results = preprocessor.process_all_patients(
        process_rtstruct=True
    )
    
    # Example 2: Process with cropping around specific structures
    results = preprocessor.process_all_patients(
        process_rtstruct=True,
        crop_to_structures=['Parotid_L', 'Parotid_R', 'Submandibular'],
        crop_margin_mm=(15.0, 15.0, 15.0)  # 15mm margin in each direction
    )
    
    # Save results
    results.to_csv(SAVE_DIR / 'processing_results.csv', index=False)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()