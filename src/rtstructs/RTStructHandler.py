import pydicom
import numpy as np
import os
import cv2
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from src.rtstructs.RTStructBuilder import RTStructBuilder
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
        self.rtstruct_ds: Optional[pydicom.Dataset] = None  # raw pydicom dataset (if needed for saving)
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
        datasets_list = []
        for root, dirs, files in os.walk(rtstructs_folder_dir):
            for file in files:
                filepath = os.path.join(root, file)

                if self.verbose:
                    print(f"  Checking file: {filepath}...")
                ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                ds = self.preprocess_ROI_names(ds, is_DLC=is_DLC)

                RTStructBuilder.validate_rtstruct(ds)
                if fix_uids:
                    RTStructBuilder.fix_rtstruct_references(ds, self.CT_series, verbose=self.verbose)
                RTStructBuilder.validate_rtstruct_series_references(ds, self.CT_series, warn_only=True)

                datasets_list.append(ds)
        
        if not datasets_list:
            raise ValueError(f"No RTSTRUCT DICOM files found in {rtstructs_folder_dir}")
        
        self.merge_rtstructs(datasets_list)

    def save_merged_rtstruct(self, output_path: str) -> str:
        """
        Save the merged RTSTRUCT dataset to disk.

        Args:
            output_path: Destination .dcm path

        Returns:
            Path where the RTSTRUCT was written
        """
        if self.rtstruct_ds is None:
            raise ValueError("No merged RTSTRUCT available. Run load_rtstructs() first.")

        output_path = str(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        pydicom.dcmwrite(output_path, self.rtstruct_ds)

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
    



    def _build_transform_matrix(self) -> np.ndarray:
        """
        Build patient-to-pixel 4x4 transformation matrix from CT series.
        Computed once and reused for all ROI rasterisation.
        """
        series_data = self.CT_series
        first_slice = series_data[0]

        offset = np.array([float(x) for x in first_slice.ImagePositionPatient])
        row_spacing, col_spacing = (float(x) for x in first_slice.PixelSpacing)

        orientation = [float(x) for x in first_slice.ImageOrientationPatient]
        row_dir = np.array(orientation[:3])
        col_dir = np.array(orientation[3:])
        slice_dir = np.cross(row_dir, col_dir)

        if len(series_data) > 1:
            first_pos = np.dot(slice_dir, [float(x) for x in series_data[0].ImagePositionPatient])
            last_pos  = np.dot(slice_dir, [float(x) for x in series_data[-1].ImagePositionPatient])
            slice_spacing = (last_pos - first_pos) / (len(series_data) - 1)
        else:
            slice_spacing = 1.0

        # Inverse of the pixel-to-patient matrix
        linear = np.zeros((3, 3), dtype=np.float64)
        linear[0, :] = row_dir   / row_spacing
        linear[1, :] = col_dir   / col_spacing
        linear[2, :] = slice_dir / slice_spacing

        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = linear
        mat[:3, 3]  = offset.dot(-linear.T)
        return mat

    def _rasterize_contours(
        self,
        ds: pydicom.Dataset,
        roi_names: List[str],
        transform: np.ndarray,
        uid_to_idx: Dict[str, int],
    ) -> Dict[str, np.ndarray]:
        """
        Rasterize RTSTRUCT contours to 3-D boolean masks (pure pydicom + cv2).

        All contours are grouped by (ROI, slice) in a single O(C) pass, then
        rasterised with cv2.fillPoly — identical algorithm to rt_utils but
        without per-ROI matrix recomputation or O(N*M) slice matching.

        Returns:
            dict mapping roi_name -> bool mask of shape (Columns, Rows, num_slices)
        """
        num_slices = len(self.CT_series)
        cols = int(self.CT_series[0].Columns)
        rows = int(self.CT_series[0].Rows)

        roi_names_set = set(roi_names)

        # Map ROI number -> name (only for structures we want)
        roi_num_to_name: Dict[int, str] = {}
        for roi in ds.StructureSetROISequence:
            if roi.ROIName in roi_names_set:
                roi_num_to_name[int(roi.ROINumber)] = roi.ROIName

        # Group contours:  roi_name -> { slice_idx -> [contour_data, ...] }
        contours_by_roi: Dict[str, Dict[int, list]] = defaultdict(lambda: defaultdict(list))

        for roi_contour in ds.ROIContourSequence:
            roi_num = int(roi_contour.ReferencedROINumber)
            if roi_num not in roi_num_to_name:
                continue
            name = roi_num_to_name[roi_num]
            if not hasattr(roi_contour, 'ContourSequence'):
                continue

            for contour in roi_contour.ContourSequence:
                if not hasattr(contour, 'ContourImageSequence') or not contour.ContourImageSequence:
                    continue
                uid = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
                if uid in uid_to_idx:
                    contours_by_roi[name][uid_to_idx[uid]].append(contour.ContourData)

        # Rasterize each ROI
        masks: Dict[str, np.ndarray] = {}
        for name in roi_names:
            if name not in contours_by_roi:
                continue

            mask = np.zeros((cols, rows, num_slices), dtype=bool)

            for slice_idx, contour_list in contours_by_roi[name].items():
                polygons = []
                for contour_data in contour_list:
                    pts = np.array(contour_data, dtype=np.float64).reshape(-1, 3)
                    aug = np.column_stack((pts, np.ones(len(pts))))
                    pixel = (aug @ transform.T)[:, :2]
                    polygons.append(np.around(pixel).astype(np.int32))

                slice_mask = np.zeros((cols, rows), dtype=np.uint8)
                cv2.fillPoly(img=slice_mask, pts=polygons, color=1)
                mask[:, :, slice_idx] = slice_mask.astype(bool)

            masks[name] = mask

        return masks

    def merge_rtstructs(
        self,
        rtstruct_datasets: List[pydicom.Dataset],
    ):
        """
        Merge multiple RTSTRUCT datasets into self.structure_masks_dict.

        For each ROI in the config structures_list, keeps the contour with the
        largest volume across all input RTSTRUCTs.

        Uses pure pydicom + cv2 for contour rasterisation (no rt_utils).
        The patient-to-pixel transformation matrix and the UID-to-index lookup
        are computed once and reused for every dataset / ROI.
        """
        if not rtstruct_datasets:
            raise ValueError("No RTSTRUCT datasets provided for merging")

        if self.verbose:
            print(f"\nMerging {len(rtstruct_datasets)} RTSTRUCT datasets...")

        structures_list = self.config['rtstruct']['structures_list']
        structures_set  = set(structures_list)

        # Geometry (computed once)
        ct_geom = self.get_ct_geometry()
        voxel_volume_mm3 = ct_geom['spacing'][0] * ct_geom['spacing'][1] * ct_geom['spacing'][2]
        transform  = self._build_transform_matrix()
        uid_to_idx = {s.SOPInstanceUID: i for i, s in enumerate(self.CT_series)}

        structure_masks: Dict[str, np.ndarray] = {}
        best_volumes:    Dict[str, float]     = {}
        self.contour_volumes = {s: 0 for s in structures_list}

        for idx, ds in enumerate(rtstruct_datasets):
            ds_roi_names = {roi.ROIName for roi in ds.StructureSetROISequence}
            needed = list(structures_set & ds_roi_names)

            if self.verbose:
                print(f"\n  RTSTRUCT {idx+1}/{len(rtstruct_datasets)}: {len(needed)} matching ROIs")

            if not needed:
                continue

            # Rasterize all needed ROIs from this dataset in one pass
            masks = self._rasterize_contours(ds, needed, transform, uid_to_idx)

            for name, mask in masks.items():
                num_voxels  = np.count_nonzero(mask)
                volume_mm3  = num_voxels * voxel_volume_mm3

                if self.verbose:
                    print(f"    - '{name}': volume = {volume_mm3:.2f} mm³ ({num_voxels} voxels)")

                if volume_mm3 > best_volumes.get(name, 0):
                    best_volumes[name] = volume_mm3
                    structure_masks[name] = np.transpose(mask, (2, 0, 1))
                    self.contour_volumes[name] = volume_mm3

        self.structure_masks_dict = structure_masks

        if self.verbose:
            num_found = sum(1 for v in self.contour_volumes.values() if v > 0)
            print(f"\n  Merge complete:")
            print(f"  - Structures found: {num_found}/{len(structures_list)}")
            print(f"  - ROIs: {list(self.structure_masks_dict.keys())}")
            print(f"  - Missing: {[s for s in structures_list if s not in self.structure_masks_dict]}")




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
    

