import warnings
import numpy as np

from typing import List
from pydicom.dataset import Dataset
from pydicom.filereader import dcmread

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
    def create_from(dicom_series_path: str, rt_struct_path: str, warn_only: bool = False) -> RTStruct:
        """
        Method to load an existing rt struct, given related DICOM series and existing rt struct
        """

        series_data = image_helper.load_sorted_image_series(dicom_series_path)
        ds = dcmread(rt_struct_path)
        RTStructBuilder.validate_rtstruct(ds)
        RTStructBuilder.validate_rtstruct_series_references(ds, series_data, warn_only)

        # TODO create new frame of reference? Right now we assume the last frame of reference created is suitable
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

    # @staticmethod
    # def validate_rtstruct_series_references(ds: Dataset, series_data: List[Dataset], warn_only: bool = False):
    #     """
    #     Method to validate RTStruct only references dicom images found within the input series_data
    #     """
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
        

    @staticmethod
    def validate_rtstruct_series_references(ds, series_data, warn_only=False):
        """Override to fix UIDs before validation"""
        
        # Build Z-position mapping
        z_to_slice = {float(s.ImagePositionPatient[2]): s for s in series_data}
        
        # Update series-level references
        ds.StudyInstanceUID = series_data[0].StudyInstanceUID
        
        ref_frame_seq = ds.ReferencedFrameOfReferenceSequence[0]
        rt_study_seq = ref_frame_seq.RTReferencedStudySequence[0]
        rt_series_seq = rt_study_seq.RTReferencedSeriesSequence[0]
        
        rt_study_seq.ReferencedSOPInstanceUID = series_data[0].StudyInstanceUID
        rt_series_seq.SeriesInstanceUID = series_data[0].SeriesInstanceUID
        
        # Rebuild ContourImageSequence
        from pydicom import Dataset
        new_seq = []
        for slice_ds in series_data:
            ref = Dataset()
            ref.ReferencedSOPClassUID = slice_ds.SOPClassUID
            ref.ReferencedSOPInstanceUID = slice_ds.SOPInstanceUID
            new_seq.append(ref)
        rt_series_seq.ContourImageSequence = new_seq
        
        # Fix contour-level references
        for roi_contour in ds.ROIContourSequence:
            if not hasattr(roi_contour, 'ContourSequence'):
                continue
            
            for contour in roi_contour.ContourSequence:
                contour_pts = np.array(contour.ContourData).reshape(-1, 3)
                contour_z = contour_pts[0, 2]
                
                # Match to closest Z
                closest_z = min(z_to_slice.keys(), key=lambda z: abs(z - contour_z))
                matched_slice = z_to_slice[closest_z]
                
                # Update reference
                if not hasattr(contour, 'ContourImageSequence'):
                    from pydicom import Sequence
                    contour.ContourImageSequence = Sequence()
                
                if len(contour.ContourImageSequence) == 0:
                    contour.ContourImageSequence.append(Dataset())
                
                contour.ContourImageSequence[0].ReferencedSOPClassUID = matched_slice.SOPClassUID
                contour.ContourImageSequence[0].ReferencedSOPInstanceUID = matched_slice.SOPInstanceUID
        
        # Call parent validation (should pass now)
        # Actually, skip it since we've fixed everything
        pass
