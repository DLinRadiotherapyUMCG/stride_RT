import warnings
from bisect import bisect_left

from typing import Dict, List
from pydicom.dataset import Dataset
from pydicom.filereader import dcmread
from pydicom import Sequence

from rt_utils.utils import SOPClassUID
from rt_utils import ds_helper, image_helper
from rt_utils.rtstruct import RTStruct


class RTStructBuilder:
    """
    Class to help facilitate the two ways in one can instantiate the RTStruct wrapper
    """

    @staticmethod
    def create_new(dicom_series_path: str) -> RTStruct:
        """
        Method to generate a new rt struct from a DICOM series
        """
        series_data = image_helper.load_sorted_image_series(dicom_series_path)
        ds = ds_helper.create_rtstruct_dataset(series_data)
        return RTStruct(series_data, ds)

    @staticmethod
    def create_from(dicom_series_path: str, rt_struct_path: str, warn_only: bool = False, 
                    fix_missing_uids: bool = True) -> RTStruct:
        """
        Method to load an existing rt struct, given related DICOM series and existing rt struct
        
        Parameters:
        -----------
        dicom_series_path : str
            Path to the directory containing the DICOM CT series
        rt_struct_path : str
            Path to the RTSTRUCT file
        warn_only : bool
            If True, issues warnings instead of raising exceptions for validation failures
        fix_missing_uids : bool
            If True, automatically fix mismatched UIDs by matching contours to slices based on Z-position
        """
        series_data = image_helper.load_sorted_image_series(dicom_series_path)
        ds = dcmread(rt_struct_path)
        RTStructBuilder.validate_rtstruct(ds)
        
        if fix_missing_uids:
            # Fix UIDs before validation
            RTStructBuilder.fix_rtstruct_references(ds, series_data)
            # After fixing, validation should pass
            RTStructBuilder.validate_rtstruct_series_references(ds, series_data, warn_only=True)
        else:
            # Original validation behavior
            RTStructBuilder.validate_rtstruct_series_references(ds, series_data, warn_only)

        return RTStruct(series_data, ds)

    @staticmethod
    def validate_rtstruct(ds: Dataset):
        """
        Method to validate a dataset is a valid RTStruct containing the required fields
        """
        if (
            ds.SOPClassUID != SOPClassUID.RTSTRUCT
            or not hasattr(ds, "ROIContourSequence")
            or not hasattr(ds, "StructureSetROISequence")
            or not hasattr(ds, "RTROIObservationsSequence")
        ):
            raise Exception("Please check that the existing RTStruct is valid")

    @staticmethod
    def fix_rtstruct_references(ds: Dataset, series_data: List[Dataset]):
        """
        Fix RTSTRUCT contour references to match the provided CT series by Z-position matching.
        This only updates `ReferencedSOPInstanceUID` values and leaves
        `ReferencedSOPClassUID` unchanged.
        """
        print("Fixing RTSTRUCT references to match CT series...")
        
        # Build Z-position to slice mapping (rounded to 1 decimal for matching)
        z_to_slice = {}
        for s in series_data:
            z_pos = round(float(s.ImagePositionPatient[2]), 1)
            z_to_slice[z_pos] = s

        # Fix contour-level references by matching Z-positions
        contours_fixed = 0
        contours_failed = 0
        
        for roi_contour in ds.ROIContourSequence:
            if not hasattr(roi_contour, 'ContourSequence'):
                continue
            
            for contour in roi_contour.ContourSequence:
                try:
                    # Extract Z-coordinate from contour data
                    contour_data = np.array(contour.ContourData).reshape(-1, 3)
                    contour_z = round(float(contour_data[0, 2]), 1)
                    
                    # Find closest matching Z-position in CT series
                    if contour_z in z_to_slice:
                        print(contour_z, "exact match found in CT series")
                        matched_slice = z_to_slice[contour_z]
                    else:
                        # Find closest Z if exact match not found
                        closest_z = min(z_to_slice.keys(), key=lambda z: abs(z - contour_z))
                        matched_slice = z_to_slice[closest_z]
                        z_diff = abs(closest_z - contour_z)
                        if z_diff > 0.5:  # Warn if difference is significant
                            warnings.warn(f"Contour Z={contour_z} matched to slice Z={closest_z} (diff={z_diff:.2f}mm)")
                    
                    # Update or create ContourImageSequence
                    if not hasattr(contour, 'ContourImageSequence'):
                        contour.ContourImageSequence = Sequence()
                    
                    if len(contour.ContourImageSequence) == 0:
                        contour.ContourImageSequence.append(Dataset())
                    
                    # Set the correct Instance UID only
                    contour.ContourImageSequence[0].ReferencedSOPInstanceUID = matched_slice.SOPInstanceUID
                    
                    contours_fixed += 1
                    
                except Exception as e:
                    contours_failed += 1
                    warnings.warn(f"Failed to fix contour reference: {str(e)}")
        
        print(f"Fixed {contours_fixed} contour references, {contours_failed} failed")


    @staticmethod
    def validate_rtstruct_series_references(ds: Dataset, series_data: List[Dataset], warn_only: bool = False):
        """
        Method to validate RTStruct only references dicom images found within the input series_data
        """
        # Validate that RTStruct references match the series data
        series_data_patient_id = series_data[0].PatientID if series_data else None
        series_data_frame_of_ref_uid = series_data[0].FrameOfReferenceUID if series_data else None
        
        if ds.PatientID != series_data_patient_id:
            msg = f"RTStruct Patient ID ({ds.PatientID}) does not match series data Patient ID ({series_data_patient_id})"
            raise Exception(msg)
        
        if hasattr(ds, 'ReferencedFrameOfReferenceSequence') and ds.ReferencedFrameOfReferenceSequence:
            rtstruct_frame_of_ref_uid = ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
            if rtstruct_frame_of_ref_uid != series_data_frame_of_ref_uid:
                msg = f"RTStruct Frame of Reference UID ({rtstruct_frame_of_ref_uid}) does not match series data Frame of Reference UID ({series_data_frame_of_ref_uid})"
                if warn_only:
                    warnings.warn(msg)
                else:
                    raise Exception(msg)
                
    #     for refd_frame_of_ref in ds.ReferencedFrameOfReferenceSequence:
    #         # Study sequence references are optional so return early if it does not exist
    #         if "RTReferencedStudySequence" not in refd_frame_of_ref:
    #             return

    #         for rt_refd_study in refd_frame_of_ref.RTReferencedStudySequence:
    #             for rt_refd_series in rt_refd_study.RTReferencedSeriesSequence:
    #                 for contour_image in rt_refd_series.ContourImageSequence:
    #                     RTStructBuilder.validate_contour_image_in_series_data(
    #                         contour_image, series_data, warn_only
    #                     )

    # @staticmethod
    # def validate_contour_image_in_series_data(
    #     contour_image: Dataset, series_data: List[Dataset], warning_only: bool = False
    # ):
    #     """
    #     Method to validate that the ReferencedSOPInstanceUID of a given contour image exists within the series data
    #     """
    #     for series in series_data:
    #         if contour_image.ReferencedSOPInstanceUID == series.SOPInstanceUID:
    #             return

    #     # ReferencedSOPInstanceUID is NOT available
    #     msg = f"Loaded RTStruct references image(s) that are not contained in input series data. " \
    #           f"Problematic image has SOP Instance Id: {contour_image.ReferencedSOPInstanceUID}"
    #     if warning_only:
    #         warnings.warn(msg)
    #     else:
    #         raise Exception(msg)
            
            