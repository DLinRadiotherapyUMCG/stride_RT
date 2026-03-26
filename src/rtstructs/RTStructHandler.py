import pydicom
import numpy as np
import os
from typing import List, Dict, Optional, Tuple

from src.rtstructs.RTStructBuilder import RTStructBuilder

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
        self.rtstruct: Optional[RTStruct] = None
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
    
    def load_rtstructs(self, rtstructs_folder_dir: str, is_DLC: bool = False, fix_uids: bool = False) -> pydicom.Dataset:
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
                if fix_uids:
                    RTStructBuilder.fix_rtstruct_references(ds, self.CT_series)
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

        merged_rtstruct = self.merge_rtstructs(rtstructs_list)

        self.rtstruct = merged_rtstruct
        # return merged_rtstruct

        # # also re-create the rtstruct_image to store in the handler for later use
        # self.rtstruct_image = self.segmentation_array_to_sitk(rtstruct_array, self.CT_image)

    def save_merged_rtstruct(self, output_path: str) -> str:
        """
        Save the merged RTSTRUCT object to disk.

        Args:
            output_path: Destination .dcm path

        Returns:
            Path where the RTSTRUCT was written
        """
        if self.rtstruct is None:
            raise ValueError("No merged RTSTRUCT available. Run load_rtstructs() first.")

        output_path = str(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if hasattr(self.rtstruct, 'save'):
            self.rtstruct.save(output_path)
        elif hasattr(self.rtstruct, 'ds') and hasattr(self.rtstruct.ds, 'save_as'):
            self.rtstruct.ds.save_as(output_path)
        elif hasattr(self.rtstruct, 'ds'):
            pydicom.dcmwrite(output_path, self.rtstruct.ds)
        else:
            raise TypeError("Unable to save RTSTRUCT: unsupported RTStruct object interface")

        if self.verbose:
            print(f"Saved merged RTSTRUCT to: {output_path}")

        return output_path
        


    

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
        structures_to_keep_set = set(structures_to_keep)
        rois_removed = []
        rois_found = []
        roi_numbers_to_remove = set()

        for roi in list(rtstruct_dataset.StructureSetROISequence):
            if roi.ROIName not in structures_to_keep_set:
                rois_removed.append(roi.ROIName)
                roi_numbers_to_remove.add(roi.ROINumber)
            else:
                rois_found.append(roi.ROIName)

        if roi_numbers_to_remove:
            rtstruct_dataset.StructureSetROISequence = [
                roi for roi in rtstruct_dataset.StructureSetROISequence
                if roi.ROINumber not in roi_numbers_to_remove
            ]
            rtstruct_dataset.ROIContourSequence = [
                rc for rc in rtstruct_dataset.ROIContourSequence
                if rc.ReferencedROINumber not in roi_numbers_to_remove
            ]
            rtstruct_dataset.RTROIObservationsSequence = [
                obs for obs in rtstruct_dataset.RTROIObservationsSequence
                if obs.ReferencedROINumber not in roi_numbers_to_remove
            ]

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
    ) -> RTStruct:
        """
        Merge multiple RTSTRUCT datasets into a single RTStruct object.
        
        For each ROI in the priority list, keeps the contour with the largest volume
        across all input RTSTRUCTs. Overlap resolution follows priority order: lower
        label index claims voxels first.
        
        Args:
            rtstruct_builders: List of RTStructBuilder objects to merge
            
        Returns:
            merged_rtstruct: RTStruct containing merged contours
        """
        if not rtstruct_builders:
            raise ValueError("No RTSTRUCT datasets provided for merging")
        
        if self.verbose:
            print(f"\nMerging {len(rtstruct_builders)} RTSTRUCT datasets...")
        
        structures_list = self.config['rtstruct']['structures_list']
        structures_set = set(structures_list)
        
        # Get CT geometry for volume calculations
        ct_geom = self.get_ct_geometry()
        pixel_spacing = ct_geom['spacing'][1:3]  # (y, x) spacing
        slice_thickness = ct_geom['spacing'][0]  # z spacing
        voxel_volume_mm3 = pixel_spacing[0] * pixel_spacing[1] * slice_thickness

        # Track selected best mask per structure (before overlap handling)
        structure_masks: Dict[str, np.ndarray] = {}
        best_volumes: Dict[str, float] = {}
        
        # Dictionary to store volume info
        self.contour_volumes = {s: 0 for s in structures_list}

        # Iterate RTSTRUCTs first: get roi_names once per file, only rasterise needed masks
        for idx, rtstruct in enumerate(rtstruct_builders):
            roi_names_set = set(rtstruct.get_roi_names())
            # Only process structures we care about that exist in this RTSTRUCT
            structures_in_this = structures_set & roi_names_set

            if self.verbose:
                print(f"\n  RTSTRUCT {idx+1}/{len(rtstruct_builders)}: {len(structures_in_this)} matching ROIs")

            for structure_name in structures_in_this:
                try:
                    roi_mask = rtstruct.get_roi_mask_by_name(structure_name)
                    num_voxels = np.count_nonzero(roi_mask)
                    volume_mm3 = num_voxels * voxel_volume_mm3
                    
                    if self.verbose:
                        print(f"    - '{structure_name}': volume = {volume_mm3:.2f} mm³ ({num_voxels} voxels)")
                    
                    # Keep track of best (largest) version
                    if volume_mm3 > best_volumes.get(structure_name, 0):
                        best_volumes[structure_name] = volume_mm3
                        structure_masks[structure_name] = np.transpose(roi_mask, (2, 0, 1))
                        self.contour_volumes[structure_name] = volume_mm3
                        
                except Exception as e:
                    if self.verbose:
                        print(f"    - '{structure_name}': Error: {e}")
                    continue


        self.structure_masks_dict = structure_masks
        """
                
        # Build merged RTSTRUCT from selected masks (independent path)
        merged_ds = ds_helper.create_rtstruct_dataset(self.CT_series)
        merged_rtstruct = RTStruct(self.CT_series, merged_ds)
        #occupied_mask_rtstruct = np.zeros(segmentation_shape, dtype=bool)

        for structure_name in structures_list:
            if structure_name in structure_masks:
                write_mask_rtstruct = structure_masks[structure_name] # & (~occupied_mask_rtstruct)
            else:
                write_mask_rtstruct = None

            if write_mask_rtstruct is not None and np.any(write_mask_rtstruct):
                merged_rtstruct.add_roi(
                    mask=write_mask_rtstruct,
                    name=structure_name
                )
                # occupied_mask_rtstruct |= write_mask_rtstruct

        """

        # Build segmentation array from selected masks (independent path)
        # this is for the DL models
        # self.rtstruct_array = self._selected_masks_to_segmentation_array(selected_masks, segmentation_shape)

        if self.verbose:
            num_structures_found = sum(1 for v in self.contour_volumes.values() if v > 0)
            print(f"\n✓ Merge complete:")
            print(f"  - Structures found: {num_structures_found}/{len(structures_list)}")
            print(f"  - ROIs in merged RTSTRUCT: {self.structure_masks_dict.keys()}")
            print(f"  - Missing ROIs: {[s for s in structures_list if s not in self.structure_masks_dict]}")

        # # Build merged RTSTRUCT from selected masks
        # merged_ds = ds_helper.create_rtstruct_dataset(self.CT_series)
        # merged_rtstruct = RTStruct(self.CT_series, merged_ds)

        # for structure_name in structures_list:
        #     if structure_name in structure_masks:
        #         write_mask_rtstruct = structure_masks[structure_name]
        #         if np.any(write_mask_rtstruct):
        #             merged_rtstruct.add_roi(
        #                 mask=write_mask_rtstruct,
        #                 name=structure_name
        #             )

        # return merged_rtstruct



    def _selected_masks_to_segmentation_array(
        self,
        struct_masks_dict: Dict[str, np.ndarray],
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
        priority_list = self.config['rtstruct']['DL_structures_list']
        segmentation_shape = next(iter(struct_masks_dict.values())).shape  # Get shape from any mask (all should be the same)

        segmentation_array = np.zeros(segmentation_shape, dtype=np.uint8)
        occupied = np.zeros(segmentation_shape, dtype=bool)

        for label_id, structure_name in enumerate(priority_list, start=1):
            if structure_name not in struct_masks_dict:
                continue
            write_mask = struct_masks_dict[structure_name] & (~occupied)
            if np.any(write_mask):
                segmentation_array[write_mask] = label_id
                occupied |= write_mask

        # Reorder axes to (z, y, x) orientation
        # segmentation_array = np.transpose(segmentation_array, (2, 1, 0))
        # segmentation_array = np.swapaxes(segmentation_array, 1, 2)
        segmentation_array = np.transpose(segmentation_array, (2, 0, 1))

        if self.verbose:
            print(f"  - Segmentation shape: {segmentation_array.shape}")
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
        Convert multi-class segmentation numpy array to SimpleITK Image.
        
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
    

