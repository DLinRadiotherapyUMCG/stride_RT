import json


def get_dicom_metadata(CT_series):
    # Get metadata from the first slice (assuming all slices have the same metadata)
    headers = json.loads(CT_series[0].to_json())

    metadata = {
        # Patient information
        'PatientID': headers['00100020']['Value'][0],
        'ScanDate': headers['00080020']['Value'][0],

        # Scanner info
        'Manufacturer': headers['00080070']['Value'][0],  # manufacturer of CT scanner
        'ModelName': headers['00081090']['Value'][0],     # model name of CT scanner
        'KiloVoltPeak': headers['00180060']['Value'][0],  # kVp used for CT acquisition

        # Image information
        'ImageOrientationPatient': headers['00200037']['Value']  + [0, 0, 1] ,  # direction
        'ImagePositionPatient': headers['00200032']['Value'],    # origin
        
        'Spacing': headers['00280030']['Value'],          # xy_spacing
        #'SliceThickness': headers['00180050']['Value'][0],
        'RescaleSlope': headers['00281053']['Value'][0],     # slope
        'RescaleIntercept': headers['00281052']['Value'][0], # intercept
    }

    n_slices = len(CT_series)
    z_min_max = [CT_series[0].ImagePositionPatient[2], CT_series[-1].ImagePositionPatient[2]]
    z_spacing = (z_min_max[-1] - z_min_max[0]) / (n_slices - 1)
    z_spacing = round(z_spacing, 1)

    assert z_spacing > 0, f"Invalid z-spacing: {z_spacing}. Check the Image Position (Patient) of the first and last slices."
    
    metadata['SliceThickness'] = z_spacing



    return metadata
