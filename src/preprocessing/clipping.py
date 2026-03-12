import numpy as np

def value_clipping(array, min_value, max_value):
    """
    Clip the values in the array to be within the specified range.

    Parameters:
    array (numpy.ndarray): The input array to be clipped.
    min_value (float): The minimum value to clip to.
    max_value (float): The maximum value to clip to.

    Returns:
    numpy.ndarray: The clipped array.
    """
    return np.clip(array, min_value, max_value)
