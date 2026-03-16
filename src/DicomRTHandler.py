import pydicom
import numpy as np
import os
import glob
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from scipy.ndimage import map_coordinates

from src.rtstructs.RTStructBuilder import RTStructBuilder
from src.rtstructs.RTStructHandler import RTStructHandler
from src.utils.dicom_header import get_dicom_metadata

from rt_utils.rtstruct import RTStruct
import SimpleITK as sitk

class DicomRTHandler:
    """
    Load and align DICOM RT data (CT, RTSTRUCT, RTDOSE)
    Keeps data as pydicom datasets until final processing
    """
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose # whether to print additional information during processing (e.g. mean doses and volumes for each structure)

        # CT
        self.CT_series: List[pydicom.Dataset] = []
        self.CT_image: sitk.Image = None
        self.CT_metadata: Dict = None
        #self.CT_array 

        # RTDOSE
        self.rtdose_image: Optional[sitk.Image] = None

        # RTSTRUCT
        self.rtstruct_image: Optional[sitk.Image] = None
        self.rtstruct_array: Optional[np.ndarray] = None

        self.OAR_metadata = {}
        self.contour_volumes = {}
        
        
        # Only create arrays when needed for resampling
        # self._ct_array_cache: Optional[np.ndarray] = None
        # self._dose_array_cache: Optional[np.ndarray] = None
        # self._dose_resampled_to_ct: Optional[np.ndarray] = None
    
    def load_ct_series(self, ct_folder: str) -> List[pydicom.Dataset]:
        """
        Load CT series as pydicom datasets
        
        Args:
            ct_folder: Path to folder containing CT DICOM files
            
        Returns:
            List of sorted CT datasets
        """
        if self.verbose:
            print(f"Loading CT series from {ct_folder}...")
        
        
        
        self.CT_series = [pydicom.dcmread(dcm) for dcm in glob.glob(ct_folder + '/*')]
        self.CT_series.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        arr_ct = np.stack([s.pixel_array for s in self.CT_series], axis=0)  # axis = 0: image.shape = (z, y, x)

        # Construct metadata_ct
        metadata = get_dicom_metadata(self.CT_series)

        # Convert pixel intensity of x to Hounsfield Units (HU): y = x * slope + intercept
        arr_ct = arr_ct * metadata['RescaleSlope'] + metadata['RescaleIntercept']   # commonly: 1.0 and -1024.0, respectively
        # arr_ct.dtype: uint16 (pixel intensity: [0, ...]) --> float64 (HU: [-1024.0, ...]) --> int16 (HU: [-1024, ...])
        #arr_ct = arr_ct.astype(cfg.ct_dtype)

        # Convert to Image
        spacing_3d = metadata['Spacing'] + [metadata['SliceThickness']]
        self.CT_image = sitk.GetImageFromArray(arr_ct)
        self.CT_image.SetOrigin(metadata['ImagePositionPatient'] )
        self.CT_image.SetDirection(metadata['ImageOrientationPatient'])
        self.CT_image.SetSpacing(spacing_3d)

        # add metadata to the handler for later use
        #self.CT_image.SetMetaData('PatientID', metadata['PatientID'])

        self.CT_metadata = metadata


        #return image_ct, metadata_ct
    
    def load_rtstructs(self, rtstructs_folder_dir: str, DLC_folder_dir: str = None, fix_uids: bool = True) -> pydicom.Dataset:
        """
        Load RTSTRUCT as pydicom dataset
        
        Args:
            rtstructs_folder_dir: Path to folder containing RTSTRUCT DICOM files
            DLC_folder_dir: Path to folder containing DLC DICOM files
            fix_uids: If True, update UIDs to match loaded CT
            
        Returns:
            RTSTRUCT dataset
        """
        if self.verbose:
            print(f"Loading RTSTRUCT from {rtstructs_folder_dir}...")
        
        # load and process all of the RTSTRUCT files in the rtstructs_folder_dir
        rtstruct_handler = RTStructHandler(self.config, CT_series=self.CT_series, CT_image=self.CT_image, verbose=self.verbose)
        rtstruct_handler.load_rtstructs(rtstructs_folder_dir, is_DLC=False)
        rtstruct_array = rtstruct_handler.rtstruct_array

        unique, counts = np.unique(rtstruct_array, return_counts=True)
        #dict(zip(unique, counts))

        # if DLC_folder_dir is provided, also load the RTSTRUCT files in the DLC folder and merge them with the ones from the rtstructs_folder_dir
        if DLC_folder_dir is not None:
            if self.verbose:
                print(f"Loading DLC RTSTRUCT from {DLC_folder_dir}...")
            DLC_rtstruct_handler = RTStructHandler(self.config, CT_series=self.CT_series, CT_image=self.CT_image, verbose=self.verbose)
            DLC_rtstruct_handler.load_rtstructs(DLC_folder_dir, is_DLC=True)
            DLC_rtstruct_array = DLC_rtstruct_handler.rtstruct_array

            unique, counts = np.unique(DLC_rtstruct_array, return_counts=True)
            dict(zip(unique, counts))

            rtstruct_array = self.merge_DLCs_with_rtrstruct(rtstruct_array, DLC_rtstruct_array)


        unique, counts = np.unique(rtstruct_array, return_counts=True)
        dict(zip(unique, counts))

        self.rtstruct_array = rtstruct_array
        self.rtstruct_image = self.segmentation_array_to_sitk(rtstruct_array, self.CT_image)



    def merge_DLCs_with_rtrstruct(self, rtstruct_array: np.ndarray, DLC_rtstruct_array: np.ndarray) -> np.ndarray:
        """
        Merge the RTSTRUCT array with the DLC RTSTRUCT array, giving priority to the non-DLC contours (i.e. if there is a conflict between the two, use the DLC contour)
        If there are unique values in the DLC that are not in the normal RTSTRUCT, add those to the merged array as well (where the normal RTSTRUCT is 0, i.e. background, to not conflict with the normal contours)

        """

        # Get unique values in both arrays
        unique_rtstruct = np.unique(rtstruct_array)  # the struct values present in the normal RTSTRUCT
        unique_DLC = np.unique(DLC_rtstruct_array)   # the struct values present in the DLC 

        # Create a copy of the original RTSTRUCT array to modify
        merged_array = rtstruct_array.copy()

        # Add any unique values from the DLC that are not in the original RTSTRUCT, but only where the original RTSTRUCT is 0 (background)
        for value in unique_DLC:
            if value == 0:
                continue  # skip background
            if value not in unique_rtstruct:
                if self.verbose:
                    print(f"Adding unique DLC structure with value {value} to merged RTSTRUCT")
                merged_array[(DLC_rtstruct_array == value) & (rtstruct_array == 0)] = value

        return merged_array
        



    
    def compute_structure_mean_doses_and_volumes(self, rtstruct_arr, dose_units='cGy', volume_units='cm3'):
        # Voxel volume: spacing from SITK is (x, y, z) in mm
        spacing = self.CT_image.GetSpacing()  # (x, y, z)
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        if volume_units == 'cm3':
            voxel_volume_units = voxel_volume_mm3 / 1000.0
        elif volume_units == 'mm3':
            voxel_volume_units = voxel_volume_mm3
        else:
            raise ValueError(f"Unsupported volume units: {volume_units}")
        
        arr_dose = self.get_dose_array()

        value_to_name_dict = {idx: name for idx, name in enumerate(self.config['rtstruct']['structures_list'], start=1)}

        # value_to_name: dict mapping int value -> structure name
        # e.g. {1: 'parotid_li', 2: 'parotid_re', ...}

        self.OAR_metadata = {}

        unique_values = np.unique(rtstruct_arr)
        unique_values = unique_values[unique_values != 0]  # exclude background

        for value in unique_values:
            name = value_to_name_dict.get(int(value), f"structure_{value}")
            
            mask = (rtstruct_arr == value)
            n_voxels = mask.sum()
            
            if n_voxels == 0:
                continue
            
            dose_values  = arr_dose[mask]

            if dose_units == 'cGy':
                dose_values = dose_values   # convert Gy to cGy
            elif dose_units == 'Gy':
                dose_values = dose_values / 100.0  # convert cGy to Gy
            else:
                raise ValueError(f"Unsupported dose units: {dose_units}")

            mean_dose    = dose_values.mean()
            volume_cm3   = n_voxels * voxel_volume_units

            self.OAR_metadata[name] = {
                f"mean_dose_{dose_units}":   float(mean_dose),
                f"volume_{volume_units}":  float(volume_cm3),
                "n_voxels":    int(n_voxels)
            }
            if self.verbose:
                print(f"{name}: mean dose = {mean_dose:.2f}, volume = {volume_cm3:.2f} {volume_units}")

        return self.OAR_metadata



    def segmentation_array_to_sitk(
        self,
        seg_array: np.ndarray,
        reference_image: sitk.Image,
        dtype: int = sitk.sitkUInt8
    ) -> sitk.Image:
        """
        Convert multi-label segmentation numpy array to SimpleITK Image.
        
        Args:
            seg_array: 3D numpy array (z, y, x) with integer labels
            reference_image: Reference SimpleITK Image (CT or RTDOSE) for geometry
            dtype: Output pixel type (default: sitkUInt8)
            
        Returns:
            SimpleITK Image with segmentation and correct geometry
        """
        # Create SimpleITK image from numpy array
        seg_image = sitk.GetImageFromArray(seg_array.astype(np.uint8))
        
        # Copy geometry from reference image
        seg_image.SetSpacing(reference_image.GetSpacing())
        seg_image.SetOrigin(reference_image.GetOrigin())
        seg_image.SetDirection(reference_image.GetDirection())
        
        # Optional: cast to desired type
        if dtype != sitk.sitkUInt8:
            seg_image = sitk.Cast(seg_image, dtype)
        
        return seg_image

    
    def load_rtdose(self, rtdose_path: str) -> pydicom.Dataset:
        """
        Load RTDOSE as pydicom dataset
        
        Args:
            rtdose_path: Path to RTDOSE DICOM file
            
        Returns:
            RTDOSE dataset
        """
        if self.verbose:
            print(f"Loading RTDOSE from {rtdose_path}...")

        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        reader.SetFileName(rtdose_path)

        self.rtdose_image = reader.Execute()

        # correct the dose grid scaling by reading the value from the pydicom dataset (since SimpleITK does not read it correctly for some reason)
        rtdose_ds = pydicom.dcmread(rtdose_path)
        dose_grid_scaling = float( rtdose_ds.DoseGridScaling) * 100  # (times 100 to convert to cGy)

        # Now scale the rtdose_image by the dose grid scaling factor
        self.rtdose_image = sitk.Cast(self.rtdose_image, sitk.sitkFloat32)
        self.rtdose_image = self.rtdose_image * dose_grid_scaling

        self.resample_dose_to_ct()
        
    
    def get_ct_geometry(self) -> Dict:
        """Extract geometry information from CT datasets"""
        if not self.CT_image:
            raise ValueError("CT not loaded")
        
         
        return {
            'origin': self.CT_image.GetOrigin(),
            'spacing': self.CT_image.GetSpacing(),
            'shape': self.CT_image.GetSize(),
            #'orientation': np.array([float(x) for x in first_ct.ImageOrientationPatient])
        }
    
    def get_dose_geometry(self) -> Dict:
        """Extract geometry information from RTDOSE dataset"""
        if self.rtdose_image is None:
            raise ValueError("RTDOSE not loaded")
        
        return {
            'origin': self.rtdose_image.GetOrigin(),
            'spacing': self.rtdose_image.GetSpacing(),
            'shape': self.rtdose_image.GetSize(),
            #'orientation': np.array([float(x) for x in first_ct.ImageOrientationPatient])
        }
    
    def resample_dose_to_ct(self) -> np.ndarray:
        """
        Resample RTDOSE to match CT geometry
        Returns the resampled dose as numpy array
        """
        self.rtdose_image = sitk.Resample(
            self.rtdose_image,
            self.CT_image,
            transform=sitk.Transform(),
            interpolator=sitk.sitkLinear,
            defaultPixelValue=0,
            outputPixelType=self.rtdose_image.GetPixelID()
        )
        
        assert self.CT_image.GetSize() == self.rtdose_image.GetSize(), "Resampled dose shape does not match CT shape"
        
        #return self.rtdose_image
    


    def get_ct_array(self) -> np.ndarray:
        """
        Get CT as numpy array
        """
        # if self._ct_array_cache is not None:
        #     return self._ct_array_cache
        if not self.CT_image:
            raise ValueError("CT not loaded")
        
        ct_array = sitk.GetArrayFromImage(self.CT_image)
        #self._ct_array_cache = ct_array
        return ct_array
    
    def get_rtstruct_array(self) -> np.ndarray:
        """
        Get RTSTRUCT segmentation as numpy array
        Returns:
            RTStruct Array: shape: (num_slices, rows, columns) Values: 0=background, 1=first structure, 2=second structure, etc.
        """
        if self.rtstruct_image is None:
            raise ValueError("RTSTRUCT not loaded")
        
        rtstruct_array = sitk.GetArrayFromImage(self.rtstruct_image)
        #self._rtstruct_array_cache = rtstruct_array
        return rtstruct_array
    
    def get_dose_array(self) -> np.ndarray:
        """
        Get dose as numpy array
        
        Returns:
            Dose array (scaled by DoseGridScaling)
        """
        # if  self._dose_array_cache is not None:
        #     return self._dose_array_cache
        
        if self.rtdose_image is None:
            raise ValueError("RTDOSE not loaded")
        
        # Get original dose array
        rtdose_array = sitk.GetArrayFromImage(self.rtdose_image)
        #self._dose_array_cache = rtdose_array

        return rtdose_array
        

        
        #return dose_array
    
    
    
    def print_summary(self):
        """Print summary of loaded data"""
        print(f"\n{'='*80}")
        print("DICOM RT DATA SUMMARY")
        print(f"{'='*80}")
        
        if self.CT_image:
            print(f"\nCT Series:")
            for key, val in self.CT_metadata.items():
                print(f"  {key}: {val}")
            # ct_geom = self.get_ct_geometry()
            # print(f"\nCT Series:")
            # #print(f"  Number of slices: {len(self.ct_datasets)}")
            # print(f"  Matrix: {ct_geom['shape']}")
            # print(f"  Spacing (z,y,x): {ct_geom['spacing']} mm")
            # print(f"  Origin: {ct_geom['origin']} mm")
        
        # if self.rtstruct_image:
        #     roi_names = self.rtstruct_image.GetMetaData('ROI_NAMES').split(',') if self.rtstruct_image.HasMetaDataKey('ROI_NAMES') else []
        #     print(f"\nRT Structure Set:")
        #     print(f"  Number of ROIs: {len(roi_names)}")
        #     print(f"  ROI names: {', '.join(roi_names)}")
        
        if self.rtdose_image:
            dose_geom = self.get_dose_geometry()
            print(f"\nRT Dose:")
            print(f"  Matrix: {dose_geom['shape']}")
            print(f"  Spacing (z,y,x): {dose_geom['spacing']} mm")
            print(f"  Origin: {dose_geom['origin']} mm")
            #print(f"  Units: {self.rtdose_dataset.DoseUnits if hasattr(self.rtdose_dataset, 'DoseUnits') else 'Unknown'}")
            
            # if self._dose_resampled_to_ct is not None:
            #     print(f"  ✓ Dose resampled to CT grid: {self._dose_resampled_to_ct.shape}")


