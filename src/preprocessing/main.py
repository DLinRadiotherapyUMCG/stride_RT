import SimpleITK as sitk

#from src.preprocessing.NUMPY_STUFF import spacing_correction, determine_center_coordinates, determine_bounding_box_bounds, crop_array, value_clipping
from src.preprocessing.cropping import determine_center_coordinates, determine_bounding_box_bounds, crop_array
from src.preprocessing.clipping import value_clipping
from src.preprocessing.spacing import spacing_correction

def preprocess_numpy_data(config, handler):
    # retrieve the numpy arrays from the handler
    CT_image = handler.CT_image
    rtdose_image = handler.rtdose_image
    #rtstruct_image = handler.rtstruct_image
    rtstruct_image = handler.make_DL_structures_image()
    # merge the RTSTRUCT masks into a single image

    # ct_metadata = handler.CT_metadata

    # apply spacing correction, if enabled in the config
    if config['preprocessing']['spacing_correction']['isEnabled']:
        print(" -- applying spacing correction --")
        print("   Original spacing:", CT_image.GetSpacing())
        
        CT_image = spacing_correction(config, CT_image, is_label=False)
        rtdose_image = spacing_correction(config, rtdose_image, is_label=False)
        rtstruct_image = spacing_correction(config, rtstruct_image, is_label=True)

        print("   Corrected spacing:", CT_image.GetSpacing())

    CT_array = sitk.GetArrayFromImage(CT_image)
    dose_array = sitk.GetArrayFromImage(rtdose_image)
    rtstruct_array = sitk.GetArrayFromImage(rtstruct_image)

    # apply cropping, if enabled in the config
    if config['preprocessing']['cropping']['isEnabled']:
        
        cropping_regions_dict = determine_center_coordinates(config, rtstruct_array)
        bounding_box_dict = determine_bounding_box_bounds(config, cropping_regions_dict)

        CT_array = crop_array(CT_array, bounding_box_dict)
        rtstruct_array = crop_array(rtstruct_array, bounding_box_dict)
        dose_array = crop_array(dose_array, bounding_box_dict)

    if config['preprocessing']['clipping']['isEnabled']:
        a_min = config['preprocessing']['clipping']['ct_min']
        a_max = config['preprocessing']['clipping']['ct_max']
        
        CT_array = value_clipping(CT_array, a_min, a_max)

    
    return CT_array, rtstruct_array, dose_array  