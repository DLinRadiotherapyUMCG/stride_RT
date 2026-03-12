import SimpleITK as sitk

def resample_to_reference(moving_image, reference_image, interpolator=sitk.sitkLinear):
    """
    Resample moving image to match reference image grid.
    
    Args:
        moving_image: SimpleITK image to resample (e.g., RTDOSE)
        reference_image: SimpleITK reference image (e.g., CT)
        interpolator: Interpolation method
        
    Returns:
        resampled_image: SimpleITK image resampled to reference grid
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(moving_image.GetPixelID())
    
    resampled_image = resampler.Execute(moving_image)
    return resampled_image