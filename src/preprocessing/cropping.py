import SimpleITK as sitk
import numpy as np



def get_segmentation_mask(segmentation_map, values):
    """
    Get segmentation mask (i.e. binary) with same shape as input segmentation_map.
    Args:
        segmentation_map (Numpy array): segmentation map
        values (list):
    Returns:
    """
    # Initialize mask
    mask = None
    for v in values:
        mask_v = (segmentation_map == v)
        if mask is None:
            mask = mask_v
        else:
            mask += mask_v

    return mask



def get_axis_bounding_box_bounds(center, size):
    """
    For single dimension: get bounding box' upper and lower bound.

    Args:
        center (int): center coordinate value of the single dimension
        size (int): total length of the bounding box in the single dimension

    Returns:
        bb_upper (int): upper bound coordinate of bounding box
        bb_lower (int): lower bound coordinate of bounding box

    """
    # Add (size % 2 > 0) = 1 if size is not divisible by 2
    bb_upper = (center + size // 2 + (size % 2 > 0))
    bb_lower = (center - size // 2)
    return int(bb_upper), int(bb_lower)


def determine_center_coordinates(config, rtstruct_array: np.ndarray) -> tuple:
    """
    Using the OARs for each of the x, y and z dimension cropping, determine the center coordinate of the bounding box
    """

    OAR_list = config['rtstruct']['structures_list']
    cropping_regions_dict = {}

    for axis_idx, direction in enumerate(['z', 'x', 'y']):
        # determine the OARs that are used for cropping in this direction
        OARs_in_direction = config['preprocessing']['cropping']['cropping_structs'][direction]
        # get the indices of the OARs in this direction, using their index in structures_list
        OAR_values = [OAR_list.index(OAR) + 1 for OAR in OARs_in_direction] # add 1 because the OAR values in the rtstruct image start from 1, not 0

        print(f"OARs in {direction} direction: {OARs_in_direction}")
        print(f"OAR values in {direction} direction: {OAR_values}")

        axis_mask = get_segmentation_mask(rtstruct_array, OAR_values)

        upper_coord = np.where(axis_mask > 0)[axis_idx].max()
        lower_coord = np.where(axis_mask > 0)[axis_idx].min() 

        assert upper_coord > lower_coord, f"Upper coordinate {upper_coord} is not greater than lower coordinate {lower_coord} for direction {direction}"

        print(f"upper_coord: {upper_coord}, lower_coord: {lower_coord}")

        cropping_regions_dict[direction] = {
            'lower_coord': lower_coord,
            'upper_coord': upper_coord,
            'center': (lower_coord + upper_coord) // 2
        }

    return cropping_regions_dict




def determine_bounding_box_bounds(config, cropping_regions_dict: dict) -> dict:
    """
    Using the OARs for each of the x, y and z dimension cropping, determine the bounding box bounds for cropping.
    """
    bounding_box_dict = {}
    BB_size = config['preprocessing']['cropping']['cropping_size']
    for axis_idx, direction in enumerate(['x', 'y', 'z']):
        # get the cropping region bounds for this direction
        cropping_region = cropping_regions_dict[direction]
        # get the size of the bounding box in this direction
        axis_size = BB_size[axis_idx]
        # get the center coordinate of the cropping region in this direction
        center = cropping_region['center']
        # determine the bounding box bounds for this direction
        bb_upper, bb_lower = get_axis_bounding_box_bounds(center, axis_size)
        bounding_box_dict[direction] = {
            f'{direction}_lower': bb_lower,
            f'{direction}_upper': bb_upper
        }
    return bounding_box_dict


def crop_array(array, bounding_box_dict):
    """
    Crop the input array using the bounding box bounds in bounding_box_dict.
    """
    z_lower, z_upper = bounding_box_dict['z']['z_lower'], bounding_box_dict['z']['z_upper']
    y_lower, y_upper = bounding_box_dict['y']['y_lower'], bounding_box_dict['y']['y_upper']
    x_lower, x_upper = bounding_box_dict['x']['x_lower'], bounding_box_dict['x']['x_upper']

    cropped_array = array[z_lower:z_upper, x_lower:x_upper, y_lower:y_upper]

    return cropped_array
