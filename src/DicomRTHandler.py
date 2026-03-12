import pydicom
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from scipy.ndimage import map_coordinates

from src.rtstructs.RTStructBuilder import RTStructBuilder
from src.rtstructs.RTSTRUCT_HANDLER import RTStructHandler
from src.utils.dicom_header import get_dicom_metadata

from rt_utils.rtstruct import RTStruct
import SimpleITK as sitk

class DicomRTHandler:
    """
    Load and align DICOM RT data (CT, RTSTRUCT, RTDOSE)
    Keeps data as pydicom datasets until final processing
    """
    
    def __init__(self, config):
        self.config = config

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
        print(f"Loading CT series from {ct_folder}...")
        import glob
        import json
        
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
        print(f"\nLoading RTSTRUCT from {rtstructs_folder_dir}...")
        
        # load and process all of the RTSTRUCT files in the rtstructs_folder_dir
        rtstruct_handler = RTStructHandler(self.config, CT_series=self.CT_series, CT_image=self.CT_image)
        rtstruct_handler.load_rtstructs(rtstructs_folder_dir, is_DLC=False)
        rtstruct_array = rtstruct_handler.rtstruct_array

        unique, counts = np.unique(rtstruct_array, return_counts=True)
        dict(zip(unique, counts))

        # if DLC_folder_dir is provided, also load the RTSTRUCT files in the DLC folder and merge them with the ones from the rtstructs_folder_dir
        if DLC_folder_dir is not None:
            print(f"\nLoading DLC RTSTRUCT from {DLC_folder_dir}...")
            DLC_rtstruct_handler = RTStructHandler(self.config, CT_series=self.CT_series, CT_image=self.CT_image)
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
            print(f"{name}: mean dose = {mean_dose:.2f}, volume = {volume_cm3:.2f} {volume_units}")

        return self.OAR_metadata




    def merge_rtstructs(
        self, 
        rtstruct_builders: List[RTStructBuilder]
    ) -> np.ndarray:
        """
        Merge multiple RTSTRUCT datasets into a single multi-label segmentation array.
        
        For each ROI in the priority list, keeps the contour with the largest volume
        across all input RTSTRUCTs. Outputs a 3D numpy array where each voxel contains
        the label ID (0 = background, 1+ = structure index).
        
        Args:
            rtstruct_builders: List of RTStructBuilder objects to merge
            
        Returns:
            segmentation_array: 3D numpy array (z, y, x) with integer labels
                            Label 0 = background
                            Label 1 = first structure in priority list
                            Label 2 = second structure in priority list, etc.
            
        Example:
            # Load multiple RTSTRUCTs
            ct_path = Path("/path/to/ct/series")
            rtstruct1 = RTStructBuilder.create_from(str(ct_path), str(rtstruct_path1))
            rtstruct2 = RTStructBuilder.create_from(str(ct_path), str(rtstruct_path2))
            
            # Merge into array
            seg_array = self.merge_rtstructs([rtstruct1, rtstruct2])
            
            # seg_array.shape = (num_slices, height, width)
            # Values: 0=background, 1=first structure, 2=second structure, etc.
        """
        if not rtstruct_builders:
            raise ValueError("No RTSTRUCT datasets provided for merging")
        
        # Handle single RTSTRUCT case
        # if len(rtstruct_builders) == 1:
        #     print("Only one RTSTRUCT dataset provided, creating segmentation from it...")
        #     return self._create_segmentation_from_single_rtstruct(rtstruct_builders[0])
        
        print(f"\nMerging {len(rtstruct_builders)} RTSTRUCT datasets...")
        
        priority_order = self.config['rtstruct']['structures_list']
        
        # Get CT geometry for volume calculations and array shape
        ct_geom = self.get_ct_geometry()
        pixel_spacing = ct_geom['spacing'][1:3]  # (y, x) spacing
        slice_thickness = ct_geom['spacing'][0]  # z spacing
        voxel_volume_mm3 = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
        
        # Get shape from first RTSTRUCT (all should match the CT)
        example_mask = rtstruct_builders[0].get_roi_mask_by_name(
            rtstruct_builders[0].get_roi_names()[0]
        )
        segmentation_shape = example_mask.shape
        
        # Initialize output segmentation array
        segmentation_array = np.zeros(segmentation_shape, dtype=np.uint8)
        
        # Dictionary to store volume info
        self.contour_volumes = {}
        
        # Loop through structure names in priority order
        for label_id, structure_name in enumerate(priority_order, start=1):
            best_volume_mm3 = 0
            best_mask = None
            
            print(f"\n  [{label_id}/{len(priority_order)}] Processing '{structure_name}'...")
            
            # Check each RTSTRUCT for this structure
            for idx, rtstruct in enumerate(rtstruct_builders):
                try:
                    # Get all ROI names in this RTSTRUCT
                    roi_names = rtstruct.get_roi_names()
                    
                    # Check if this structure exists
                    if structure_name not in roi_names:
                        print(f"    - RTSTRUCT {idx+1}/{len(rtstruct_builders)}: '{structure_name}' not found")
                        continue
                    
                    # Get the 3D mask
                    roi_mask = rtstruct.get_roi_mask_by_name(structure_name)

                    # # fill holes
                    # #from scipy.ndimage import binary_fill_holes
                    # roi_mask = binary_fill_holes(roi_mask).astype(np.uint8)
                    
                    # Calculate volume
                    num_voxels = np.sum(roi_mask > 0)
                    volume_mm3 = num_voxels * voxel_volume_mm3
                    
                    print(f"    - RTSTRUCT {idx+1}/{len(rtstruct_builders)}: volume = {volume_mm3:.2f} mm³ ({num_voxels} voxels)")
                    
                    # Keep track of best version
                    if volume_mm3 > best_volume_mm3:
                        best_volume_mm3 = volume_mm3
                        best_mask = roi_mask
                        
                except Exception as e:
                    print(f"    - RTSTRUCT {idx+1}/{len(rtstruct_builders)}: Error: {e}")
                    continue
            
            # Add best mask to segmentation array
            if best_mask is not None and best_volume_mm3 > 0:
                # Assign this label ID to all voxels in the mask
                # if there are voxels in the mask which are already non-zero in the segmentation array, print a warning that there is overlap between structures
                overlap_voxels = np.sum((best_mask > 0) & (segmentation_array > 0))
                if overlap_voxels > 0:
                    print(f"      ⚠️ Warning: {overlap_voxels} voxels overlap with previously added structures. This may indicate nested or overlapping contours.")
                    # print which structures are overlapping
                    overlapping_labels = np.unique(segmentation_array[(best_mask > 0) & (segmentation_array > 0)])
                    overlapping_structures = [priority_order[label-1] for label in overlapping_labels if label > 0 and label <= len(priority_order)]
                    print(f"        Overlapping structures: {overlapping_structures}")

                # and only do so for voxels which do not already have a structure
                segmentation_array[(best_mask > 0) & (segmentation_array == 0)] = label_id
                self.contour_volumes[structure_name] = best_volume_mm3
                print(f"    ✓ Added to segmentation with label {label_id} (volume: {best_volume_mm3:.2f} mm³)")

            else:
                self.contour_volumes[structure_name] = 0
                print(f"    ✗ '{structure_name}' not found in any RTSTRUCT")
        
        # Summary
        segmentation_array = np.transpose(segmentation_array, (2, 1, 0))
        # rotate 90 degrees to match (x,y,z) orientation
        segmentation_array = np.swapaxes(segmentation_array, 1, 2)

        num_structures_found = sum(1 for v in self.contour_volumes.values() if v > 0)
        print(f"\n✓ Merge complete:")
        print(f"  - Segmentation shape: {segmentation_array.shape}")
        print(f"  - Structures found: {num_structures_found}/{len(priority_order)}")
        print(f"  - Unique labels: {np.unique(segmentation_array)}")
        print(f"  - Background voxels: {np.sum(segmentation_array == 0)}")
        print(f"  - Structure voxels: {np.sum(segmentation_array > 0)}")
        
        return segmentation_array



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
        print(f"\nLoading RTDOSE from {rtdose_path}...")

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
            ct_geom = self.get_ct_geometry()
            print(f"\nCT Series:")
            #print(f"  Number of slices: {len(self.ct_datasets)}")
            print(f"  Matrix: {ct_geom['shape']}")
            print(f"  Spacing (z,y,x): {ct_geom['spacing']} mm")
            print(f"  Origin: {ct_geom['origin']} mm")
        
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











    # def _create_segmentation_from_single_rtstruct(
    #         self, 
    #         rtstruct: RTStructBuilder
    #     ) -> np.ndarray:
    #     """
    #     Helper function to create segmentation array from a single RTSTRUCT.
        
    #     Args:
    #         rtstruct: RTStructBuilder object
            
    #     Returns:
    #         segmentation_array: 3D numpy array with labels
    #     """
    #     priority_order = self.config['rtstruct']['structures_list']
    #     roi_names = rtstruct.get_roi_names()
        
    #     # Get shape from first available ROI
    #     if not roi_names:
    #         raise ValueError("RTSTRUCT contains no ROIs")
        
    #     example_mask = rtstruct.get_roi_mask_by_name(roi_names[0])
    #     segmentation_shape = example_mask.shape
        
    #     # Initialize output array
    #     segmentation_array = np.zeros(segmentation_shape, dtype=np.uint8)
        
    #     # Get CT geometry for volume calculations
    #     ct_geom = self.get_ct_geometry()
    #     pixel_spacing = ct_geom['spacing'][1:3]
    #     slice_thickness = ct_geom['spacing'][0]
    #     voxel_volume_mm3 = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
        
    #     self.contour_volumes = {}
        
    #     # Add each structure in priority order
    #     for label_id, structure_name in enumerate(priority_order, start=1):
    #         if structure_name in roi_names:
    #             try:
    #                 roi_mask = rtstruct.get_roi_mask_by_name(structure_name)
    #                 num_voxels = np.sum(roi_mask > 0)
    #                 volume_mm3 = num_voxels * voxel_volume_mm3
                    
    #                 segmentation_array[roi_mask > 0] = label_id
    #                 self.contour_volumes[structure_name] = volume_mm3
    #                 print(f"  [{label_id}] Added '{structure_name}' (volume: {volume_mm3:.2f} mm³)")
    #             except Exception as e:
    #                 print(f"  [{label_id}] Failed to add '{structure_name}': {e}")
    #                 self.contour_volumes[structure_name] = 0
    #         else:
    #             self.contour_volumes[structure_name] = 0
        
    #     return segmentation_array


    # def _fix_rtstruct_uids(self, rtstruct_dataset):
    #     """Fix RTSTRUCT UIDs to reference loaded CT datasets"""
    #     if not self.ct_datasets:
    #         raise ValueError("CT datasets must be loaded before fixing UIDs")
        
    #     # Build mapping of Z position to CT dataset
    #     z_to_ct = {float(ds.ImagePositionPatient[2]): ds for ds in self.ct_datasets}
        
    #     first_ct = self.ct_datasets[0]
        
    #     # Update Study/Series UIDs
    #     rtstruct_dataset.StudyInstanceUID = first_ct.StudyInstanceUID
        
    #     # Update referenced frame of reference
    #     if hasattr(rtstruct_dataset, 'ReferencedFrameOfReferenceSequence'):
    #         ref_frame_seq = rtstruct_dataset.ReferencedFrameOfReferenceSequence[0]
            
    #         if hasattr(ref_frame_seq, 'RTReferencedStudySequence'):
    #             rt_study_seq = ref_frame_seq.RTReferencedStudySequence[0]
    #             rt_study_seq.ReferencedSOPInstanceUID = first_ct.StudyInstanceUID
                
    #             if hasattr(rt_study_seq, 'RTReferencedSeriesSequence'):
    #                 rt_series_seq = rt_study_seq.RTReferencedSeriesSequence[0]
    #                 rt_series_seq.SeriesInstanceUID = first_ct.SeriesInstanceUID
                    
    #                 # Rebuild ContourImageSequence with correct CT UIDs
    #                 new_contour_seq = pydicom.Sequence()
    #                 for ct_ds in self.ct_datasets:
    #                     ref_image = pydicom.Dataset()
    #                     ref_image.ReferencedSOPClassUID = ct_ds.SOPClassUID
    #                     ref_image.ReferencedSOPInstanceUID = ct_ds.SOPInstanceUID
    #                     new_contour_seq.append(ref_image)
                    
    #                 rt_series_seq.ContourImageSequence = new_contour_seq
        
    #     # Fix individual contour references by matching Z positions
    #     if hasattr(rtstruct_dataset, 'ROIContourSequence'):
    #         for roi_contour in rtstruct_dataset.ROIContourSequence:
    #             if not hasattr(roi_contour, 'ContourSequence'):
    #                 continue
                
    #             for contour in roi_contour.ContourSequence:
    #                 if not hasattr(contour, 'ContourData'):
    #                     continue
                    
    #                 # Get Z position from first point in contour
    #                 contour_data = contour.ContourData
    #                 contour_z = float(contour_data[2])  # z is at index 2
                    
    #                 # Find closest CT slice by Z position
    #                 closest_z = min(z_to_ct.keys(), key=lambda z: abs(z - contour_z))
    #                 matched_ct = z_to_ct[closest_z]
                    
    #                 # Update or create ContourImageSequence
    #                 if not hasattr(contour, 'ContourImageSequence'):
    #                     contour.ContourImageSequence = pydicom.Sequence()
                    
    #                 if len(contour.ContourImageSequence) == 0:
    #                     contour.ContourImageSequence.append(pydicom.Dataset())
                    
    #                 contour.ContourImageSequence[0].ReferencedSOPClassUID = matched_ct.SOPClassUID
    #                 contour.ContourImageSequence[0].ReferencedSOPInstanceUID = matched_ct.SOPInstanceUID
        
    #     print("  ✓ UIDs fixed")
    #     return rtstruct_dataset




    # def _resample_volume(self,
    #                     source_array: np.ndarray,
    #                     source_origin: np.ndarray,
    #                     source_spacing: np.ndarray,
    #                     target_origin: np.ndarray,
    #                     target_spacing: np.ndarray,
    #                     target_shape: Tuple[int, int, int]) -> np.ndarray:
    #     """
    #     Resample a 3D volume from source grid to target grid using trilinear interpolation
        
    #     Args:
    #         source_array: Source 3D array (z, y, x)
    #         source_origin: Source origin (x, y, z) in mm
    #         source_spacing: Source spacing (z, y, x) in mm
    #         target_origin: Target origin (x, y, z) in mm
    #         target_spacing: Target spacing (z, y, x) in mm
    #         target_shape: Target shape (z, y, x)
        
    #     Returns:
    #         Resampled 3D array
    #     """
    #     # Create coordinate grids for target volume
    #     z_coords = np.arange(target_shape[0]) * target_spacing[0] + target_origin[2]
    #     y_coords = np.arange(target_shape[1]) * target_spacing[1] + target_origin[1]
    #     x_coords = np.arange(target_shape[2]) * target_spacing[2] + target_origin[0]
        
    #     # Create meshgrid
    #     zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        
    #     # Convert world coordinates to source array indices
    #     z_indices = (zz - source_origin[2]) / source_spacing[0]
    #     y_indices = (yy - source_origin[1]) / source_spacing[1]
    #     x_indices = (xx - source_origin[0]) / source_spacing[2]
        
    #     # Stack coordinates
    #     coords = np.array([z_indices, y_indices, x_indices])
        
    #     # Perform interpolation
    #     resampled = map_coordinates(
    #         source_array,
    #         coords,
    #         order=1,  # Linear interpolation
    #         mode='constant',
    #         cval=0.0
    #     )
        
    #     return resampled
    











# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    pass
    
    # # ===== Basic usage: Load and align =====
    # handler = DicomRTHandler()
    
    # # Load data (keeps as pydicom datasets)
    # handler.load_ct_series('path/to/ct_folder')
    # handler.load_rtstruct('path/to/rtstruct.dcm', fix_uids=True)
    # handler.load_rtdose('path/to/rtdose.dcm')
    
    # # Print summary
    # handler.print_summary()
    
    # # Resample dose to CT grid
    # dose_resampled = handler.resample_dose_to_ct()
    
    # # Now you can work with arrays when needed
    # ct_array = handler.get_ct_array()
    # dose_array = handler.get_dose_array(resampled=True)
    
    # print(f"\nFinal arrays:")
    # print(f"  CT shape: {ct_array.shape}")
    # print(f"  Dose shape: {dose_array.shape}")
    # print(f"  Match: {ct_array.shape == dose_array.shape}")
    
    # # Create masks for ROIs
    # roi_names = handler.get_roi_names()
    # for roi_name in roi_names:
    #     mask = handler.create_roi_mask(roi_name)
    #     print(f"  {roi_name} mask: {mask.shape}, {np.sum(mask)} voxels")
    
    
    # # ===== Access original pydicom datasets =====
    
    # # Access individual CT slices
    # first_ct_slice = handler.ct_datasets[0]
    # print(f"\nFirst CT slice:")
    # print(f"  SOP Instance UID: {first_ct_slice.SOPInstanceUID}")
    # print(f"  Position: {first_ct_slice.ImagePositionPatient}")
    
    # # Access RTSTRUCT
    # rtstruct = handler.rtstruct_dataset
    # print(f"\nRTSTRUCT:")
    # print(f"  Structure Set Label: {rtstruct.StructureSetLabel if hasattr(rtstruct, 'StructureSetLabel') else 'N/A'}")
    
    # # Access RTDOSE
    # rtdose = handler.rtdose_dataset
    # print(f"\nRTDOSE:")
    # print(f"  Dose Type: {rtdose.DoseType if hasattr(rtdose, 'DoseType') else 'N/A'}")
    # print(f"  Dose Summation Type: {rtdose.DoseSummationType if hasattr(rtdose, 'DoseSummationType') else 'N/A'}")
    
    
    # # ===== Get specific ROI mask =====
    # ptv_mask = handler.create_roi_mask('PTV')
    # print(f"\nPTV mask shape: {ptv_mask.shape}")
    # print(f"PTV volume voxels: {np.sum(ptv_mask)}")