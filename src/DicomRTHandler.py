import pydicom
import numpy as np
import os
import glob
from typing import List, Dict, Optional, Tuple

from src.rtstructs.RTStructHandler import RTStructHandler
from src.utils.dicom_header import get_dicom_metadata
from src.DVH.dvh import compute_dvh_metrics

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
        self.struct_masks_dict: Optional[Dict[str, np.ndarray]] = None

        self.OAR_metadata = {}
        self.contour_volumes = {}
        
        
    
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
        rtstruct_handler.load_rtstructs(rtstructs_folder_dir, fix_uids=fix_uids, is_DLC=False)
        self.structure_masks_dict = rtstruct_handler.structure_masks_dict

        # if DLC_folder_dir is provided, also load the RTSTRUCT files in the DLC folder and merge them with the ones from the rtstructs_folder_dir
        if DLC_folder_dir is not None:
            if self.verbose:
                print(f"Loading DLC RTSTRUCT from {DLC_folder_dir}...")

            # Build a config with only the missing structures for the DLC handler
            dlc_config = self.config.copy()
            dlc_config['rtstruct'] = self.config['rtstruct'].copy()
            dlc_config['rtstruct']['structures_list'] = [
                x for x in self.config['rtstruct']['structures_list']
                if x not in self.structure_masks_dict
            ]
            DLC_rtstruct_handler = RTStructHandler(dlc_config, CT_series=self.CT_series, CT_image=self.CT_image, verbose=self.verbose)
            DLC_rtstruct_handler.load_rtstructs(DLC_folder_dir, fix_uids=fix_uids, is_DLC=True)
            DLC_structure_masks_dict = DLC_rtstruct_handler.structure_masks_dict
            #DLC_rtstruct_array = DLC_rtstruct_handler.rtstruct_array

            self.structure_masks_dict = self.merge_DLCs_with_rtrstruct(self.structure_masks_dict, DLC_structure_masks_dict)




    def merge_DLCs_with_rtrstruct(self, rtstruct_dict, DLC_rtstruct_dict):
        """
        Merge the RTSTRUCT array with the DLC RTSTRUCT array, giving priority to the non-DLC contours (i.e. if there is a conflict between the two, use the DLC contour)
        If there are unique values in the DLC that are not in the normal RTSTRUCT, add those to the merged array as well (where the normal RTSTRUCT is 0, i.e. background, to not conflict with the normal contours)

        """

        structures_list = self.config['rtstruct']['structures_list']

        main_structs = rtstruct_dict.keys()  # struct names already present in the main RTSTRUCT
        DLC_structs = DLC_rtstruct_dict.keys() # struct names present in the DLC RTSTRUCT 


        # if there any structs missing in main_structs, add them from DLC_structs (but only if they are in the structures_list)
        for struct in structures_list:
            if struct not in main_structs and struct in DLC_structs:
                rtstruct_dict[struct] = DLC_rtstruct_dict[struct]
                
        return rtstruct_dict
        



    
    def compute_structure_mean_doses_and_volumes(self, dose_units='cGy', volume_units='cm3'):
        # compute the mean dose and volume for each structure in the RTSTRUCT, and store it in a dictionary (self.OAR_metadata) with the structure name as key and a dictionary with mean dose and volume as value


        # Voxel volume: spacing from SITK is (x, y, z) in mm
        spacing = self.CT_image.GetSpacing()  # (x, y, z)
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        if volume_units == 'cm3':
            voxel_volume_units = voxel_volume_mm3 / 1000.0
        elif volume_units == 'mm3':
            voxel_volume_units = voxel_volume_mm3
        else:
            raise ValueError(f"Unsupported volume units: {volume_units}")
        
        rtdose_arr = self.get_dose_array()


        

        dvh_stats_dict = compute_dvh_metrics(rtdose_arr, self.structure_masks_dict, 
                                             voxel_volume_cc=voxel_volume_units, dose_units=dose_units)
        self.OAR_metadata = dvh_stats_dict

        # print the mean doses and volumes for each structure
        if self.verbose:
            print("\nStructure mean doses and volumes:")
            for struct_name in self.structure_masks_dict.keys():
                mean_dose = dvh_stats_dict[f'{struct_name}_Dmean']
                volume_cc = dvh_stats_dict[f'{struct_name}_volume']
                print(f"  {struct_name}: mean dose = {mean_dose:.2f} {dose_units}, volume = {volume_cc:.2f} {volume_units}")


        #transformed_dose_arr = np.transpose(rtdose_arr, (1, 2, 0))

        #mask = rtstruct.get_roi_mask_by_name(structures_list[0])

        # compute mean doses
        # for name in structures_list:
        #     if name in structs_present:
        #         mask = self.structure_masks_dict[name]

        #         n_voxels = mask.sum()
        #         volume_cm3   = n_voxels * voxel_volume_units

        #         # print(rtdose_arr.shape)
        #         # print(mask.shape)

        #         #mean_dose = rtdose_arr[mask].mean() 

        #         dose_values = rtdose_arr[mask]

        #         if dose_units == 'cGy':
        #             dose_values = dose_values   # convert Gy to cGy
        #         elif dose_units == 'Gy':
        #             dose_values = dose_values / 100.0  # convert cGy to Gy
        #         else:
        #             raise ValueError(f"Unsupported dose units: {dose_units}")

        #         mean_dose = dose_values.mean()

        #         """
        #         COMPUTE DVH STATS HERE!
        #         """

        #         # print(mask.shape)
        #         # print(f"{name}: mean dose = {mean_dose/1000:.2f} Gy")
        #         self.OAR_metadata[name] = {
        #             f"mean_dose_{dose_units}":   float(mean_dose),
        #             f"volume_{volume_units}":  float(volume_cm3),
        #             "n_voxels":    int(n_voxels)
        #         }
        #         if self.verbose:
        #             print(f"{name}: mean dose = {mean_dose:.2f}, volume = {volume_cm3:.2f} {volume_units}")
        #     else:
        #         print(f"{name} is NOT present in the RTSTRUCT")


        return self.OAR_metadata



    def make_DL_structures_image(
            self
        #config,
        #struct_masks_dict, #: Dict[str, np.ndarray],
        #segmentation_shape: Tuple
    ) -> np.ndarray:
        """
        Convert a dict of per-structure boolean masks into a multi-class
        segmentation array with priority-based overlap resolution.

        Args:
            struct_masks_dict: Mapping of structure name -> boolean 3-D mask
            #segmentation_shape: Shape of the output array before axis reordering

        Returns:
            segmentation_array: 3-D numpy array (z, y, x) with integer labels
        """
        structure_masks_dict = self.structure_masks_dict

        priority_list = self.config['rtstruct']['DL_structures_list']
        segmentation_shape = next(iter(structure_masks_dict.values())).shape  # Get shape from any mask (all should be the same)

        segmentation_array = np.zeros(segmentation_shape, dtype=np.uint8)

        for label_id, structure_name in enumerate(priority_list, start=1):

            if structure_name not in structure_masks_dict:
                continue
            # the write mask is the current structure, and where there already isn't a structure assigned (occupied)
            write_mask = (structure_masks_dict[structure_name] > 0) & (segmentation_array == 0)

            if np.any(write_mask):
                segmentation_array[write_mask] = label_id

        if self.verbose:
            print(f"  - Segmentation shape: {segmentation_array.shape}")
            print(f"  - Unique labels: {np.unique(segmentation_array)}")
            print(f"  - Background voxels: {np.sum(segmentation_array == 0)}")
            print(f"  - Structure voxels: {np.sum(segmentation_array > 0)}")


        segmentation_image = self.segmentation_array_to_sitk(segmentation_array, self.CT_image, dtype=sitk.sitkUInt8)

        return segmentation_image



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


