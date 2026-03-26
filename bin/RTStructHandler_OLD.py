import pydicom
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from scipy.ndimage import map_coordinates

from src.rtstructs.RTStructBuilder import RTStructBuilder
from src.utils.dicom_header import get_dicom_metadata

from rt_utils.rtstruct import RTStruct
import SimpleITK as sitk

class RTStructHandler:
    """
    Load and align DICOM RT data (CT, RTSTRUCT, RTDOSE)
    Keeps data as pydicom datasets until final processing
    """
    
    def __init__(self, config, CT_series: List[pydicom.Dataset], CT_image: sitk.Image, verbose: bool = False):
        self.config = config
        self.verbose = verbose

        # CT
        self.CT_series = CT_series
        self.CT_image = CT_image

        # RTSTRUCT
        self.rtstruct_image: Optional[sitk.Image] = None
        self.rtstruct_array: Optional[np.ndarray] = None

        self.OAR_metadata = {}
        self.contour_volumes = {}
        
        
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
    
    def load_rtstructs(self, rtstructs_folder_dir: str, is_DLC: bool = False, fix_uids: bool = True) -> pydicom.Dataset:
        """
        Load RTSTRUCT as pydicom dataset
        
        Args:
            rtstructs_folder_dir: Path to folder containing RTSTRUCT DICOM files
            is_DLC: If True, process as DLC RTSTRUCT
            fix_uids: If True, update UIDs to match loaded CT
            
        Returns:
            RTSTRUCT dataset
        """
        if self.verbose:
            print(f"Loading RTSTRUCT from {rtstructs_folder_dir}...")
        
        # Find all RTSTRUCT DICOM files in the folder
        rtstructs_list = []
        for root, dirs, files in os.walk(rtstructs_folder_dir):
            for file in files:
                filepath = os.path.join(root, file)

                #try:
                if self.verbose:
                    print(f"  Checking file: {filepath}...")
                ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                ds = self.preprocess_ROI_names(ds, is_DLC=is_DLC)         # correct the names in the rtstruct, and remove all the extra ROIs that are not in the config file (e.g. those with DLC tags, or other extra ROIs that are not needed for modelling)

                # construct the rtstruct volume
                RTStructBuilder.validate_rtstruct(ds)
                RTStructBuilder.validate_rtstruct_series_references(ds, self.CT_series, warn_only=True)
                
                rtstruct = RTStruct(self.CT_series, ds)

                rtstructs_list.append(rtstruct) 
        
        if not rtstructs_list:
            raise ValueError(f"No RTSTRUCT DICOM files found in {rtstructs_folder_dir}")
        
        # Merge the RTSTRUCT files if there are multiple (combining ROIs)
        # if len (rtstructs_list) > 1:
        #     print(f"\nMultiple RTSTRUCT files found ({len(rtstructs_list)}), merging into single segmentation array...")
        #     rtstruct_array = self.merge_rtstructs(rtstructs_list)
        # else:
        #     print(f"\nOnly one RTSTRUCT file found, creating segmentation array from it...")
        #     rtstruct_array = self._create_segmentation_from_single_rtstruct(rtstructs_list[0])

        rtstruct_array = self.merge_rtstructs(rtstructs_list)

        self.rtstruct_array = rtstruct_array

        # # also re-create the rtstruct_image to store in the handler for later use
        # self.rtstruct_image = self.segmentation_array_to_sitk(rtstruct_array, self.CT_image)
        


    

    def preprocess_ROI_names(self, rtstruct_dataset, is_DLC=False):
        """
        Correct ROI names, e.g. by removing DLC tags or applying mapping from config file
        Also remove extra ROIs
        Args:            
            rtstruct_dataset: RTSTRUCT dataset to preprocess
        Returns:
            Preprocessed RTSTRUCT dataset
        """
        mapping_dict = self.config['rtstruct']['struct_name_corrections']
        structures_to_keep = self.config['rtstruct']['structures_list']

        # Build mapping of original to corrected names
        original_to_corrected = {}
        for roi in rtstruct_dataset.StructureSetROISequence:
            original_name = roi.ROIName
            if is_DLC:
                original_name_no_DLC = self._remove_DLC_tag_from_structure_name(original_name)  # Remove DLC tag if present
                corrected_name = mapping_dict.get(original_name_no_DLC, original_name_no_DLC)  # Apply mapping if exists
            else:
                corrected_name = mapping_dict.get(original_name, original_name)  # Apply mapping if exists
            
            original_to_corrected[original_name] = corrected_name

        # Update ROI names in the dataset
        for roi in rtstruct_dataset.StructureSetROISequence:
            original_name = roi.ROIName
            roi.ROIName = original_to_corrected[original_name]

        # Remove ROIs that are not in the structures_to_keep list 
        # (makes the rest of the code faster, because we don't have to process as many ROIs later on)
        rois_removed = []
        rois_found = []
        for roi in rtstruct_dataset.StructureSetROISequence:
            if roi.ROIName not in structures_to_keep:
                rois_removed.append(roi.ROIName)

                # remove the ROI
                rtstruct_dataset.StructureSetROISequence.remove(roi)
                # Also need to remove corresponding entries in ROIContourSequence and RTROIObservationsSequence
                for rc in rtstruct_dataset.ROIContourSequence:
                    if rc.ReferencedROINumber == roi.ROINumber:
                        rtstruct_dataset.ROIContourSequence.remove(rc)
                        break
                for obs in rtstruct_dataset.RTROIObservationsSequence:
                    if obs.ReferencedROINumber == roi.ROINumber:
                        rtstruct_dataset.RTROIObservationsSequence.remove(obs)
                        break
            else:
                rois_found.append(roi.ROIName)

        if self.verbose:
            print(f"Removed ROIs: {rois_removed}")
            print(f"Kept ROIs: {rois_found}")
        

        return rtstruct_dataset

    
    def _remove_DLC_tag_from_structure_name(self, structure_name):
        DLC_tag_string = self.config['rtstruct']['DLC_name_tag']
        structure_name = structure_name.replace(DLC_tag_string, "")  # strip DLC tag from structure name if present
        return structure_name
    



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
        
        
        if self.verbose:
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
            
            if self.verbose:
                print(f"\n  [{label_id}/{len(priority_order)}] Processing '{structure_name}'...")
            
            # Check each RTSTRUCT for this structure
            for idx, rtstruct in enumerate(rtstruct_builders):
                try:
                    # Get all ROI names in this RTSTRUCT
                    roi_names = rtstruct.get_roi_names()
                    
                    # Check if this structure exists
                    if structure_name not in roi_names:
                        if self.verbose:
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
                    
                    if self.verbose:
                        print(f"    - RTSTRUCT {idx+1}/{len(rtstruct_builders)}: volume = {volume_mm3:.2f} mm³ ({num_voxels} voxels)")
                    
                    # Keep track of best version
                    if volume_mm3 > best_volume_mm3:
                        best_volume_mm3 = volume_mm3
                        best_mask = roi_mask
                        
                except Exception as e:
                    if self.verbose:
                        print(f"    - RTSTRUCT {idx+1}/{len(rtstruct_builders)}: Error: {e}")
                    continue
            
            # Add best mask to segmentation array
            if best_mask is not None and best_volume_mm3 > 0:
                # Assign this label ID to all voxels in the mask
                # if there are voxels in the mask which are already non-zero in the segmentation array, print a warning that there is overlap between structures
                overlap_voxels = np.sum((best_mask > 0) & (segmentation_array > 0))
                if overlap_voxels > 0:
                    if self.verbose:
                        print(f"      ⚠️ Warning: {overlap_voxels} voxels overlap with previously added structures. This may indicate nested or overlapping contours.")
                    # print which structures are overlapping
                    overlapping_labels = np.unique(segmentation_array[(best_mask > 0) & (segmentation_array > 0)])
                    overlapping_structures = [priority_order[label-1] for label in overlapping_labels if label > 0 and label <= len(priority_order)]
                    if self.verbose:
                        print(f"        Overlapping structures: {overlapping_structures}")

                # and only do so for voxels which do not already have a structure
                segmentation_array[(best_mask > 0) & (segmentation_array == 0)] = label_id
                self.contour_volumes[structure_name] = best_volume_mm3
                if self.verbose:
                    print(f"    ✓ Added to segmentation with label {label_id} (volume: {best_volume_mm3:.2f} mm³)")

            else:
                self.contour_volumes[structure_name] = 0
                if self.verbose:
                    print(f"    ✗ '{structure_name}' not found in any RTSTRUCT")
        
        # Summary
        segmentation_array = np.transpose(segmentation_array, (2, 1, 0))
        # rotate 90 degrees to match (x,y,z) orientation
        segmentation_array = np.swapaxes(segmentation_array, 1, 2)

        num_structures_found = sum(1 for v in self.contour_volumes.values() if v > 0)
        if self.verbose:
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

        

        
        #return dose_array
    
    
    
    def print_summary(self):
        """Print summary of loaded data"""
        print(f"\n{'='*80}")
        print("DICOM RT DATA SUMMARY")
        print(f"{'='*80}")
        
        # if self.CT_image:
        #     ct_geom = self.get_ct_geometry()
        #     print(f"\nCT Series:")
        #     #print(f"  Number of slices: {len(self.ct_datasets)}")
        #     print(f"  Matrix: {ct_geom['shape']}")
        #     print(f"  Spacing (z,y,x): {ct_geom['spacing']} mm")
        #     print(f"  Origin: {ct_geom['origin']} mm")
        
        # # if self.rtstruct_image:
        # #     roi_names = self.rtstruct_image.GetMetaData('ROI_NAMES').split(',') if self.rtstruct_image.HasMetaDataKey('ROI_NAMES') else []
        # #     print(f"\nRT Structure Set:")
        # #     print(f"  Number of ROIs: {len(roi_names)}")
        # #     print(f"  ROI names: {', '.join(roi_names)}")
        
        # if self.rtdose_image:
        #     dose_geom = self.get_dose_geometry()
        #     print(f"\nRT Dose:")
        #     print(f"  Matrix: {dose_geom['shape']}")
        #     print(f"  Spacing (z,y,x): {dose_geom['spacing']} mm")
        #     print(f"  Origin: {dose_geom['origin']} mm")
        #     #print(f"  Units: {self.rtdose_dataset.DoseUnits if hasattr(self.rtdose_dataset, 'DoseUnits') else 'Unknown'}")
            
        #     # if self._dose_resampled_to_ct is not None:
        #     #     print(f"  ✓ Dose resampled to CT grid: {self._dose_resampled_to_ct.shape}")








    


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

    #     print("CONTAINS", roi_names)
        
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
                
    #             roi_mask = rtstruct.get_roi_mask_by_name(structure_name)
    #             num_voxels = np.sum(roi_mask > 0)
    #             volume_mm3 = num_voxels * voxel_volume_mm3
                
    #             segmentation_array[(roi_mask > 0) & (segmentation_array == 0)] = label_id
    #             self.contour_volumes[structure_name] = volume_mm3
    #             print(f"  [{label_id}] Added '{structure_name}' (volume: {volume_mm3:.2f} mm³)")
                    
    #             # except Exception as e:
    #             #     print(f"  [{label_id}] Failed to add '{structure_name}': {e}")
    #             #     self.contour_volumes[structure_name] = 0
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
    

