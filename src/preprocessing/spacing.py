import SimpleITK as sitk

def spacing_correction(config, image: sitk.Image, is_label=False) -> sitk.Image:
    """
    
    """
    # resample the spacing to the desired spacing in the config (but only if the current spacing is different from the desired spacing)
    current_spacing = image.GetSpacing()
    desired_spacing = config['preprocessing']['spacing_correction']['spacing']
    perform_spacing_correction = config['preprocessing']['spacing_correction']['isEnabled']

    if perform_spacing_correction and (current_spacing != desired_spacing):
        print(f"Resampling image from spacing {current_spacing} to {desired_spacing}")
        resampler = sitk.ResampleImageFilter()

        out_size = [
            int(image.GetSize()[i] * (current_spacing[i] / desired_spacing[i]))
            for i in range(3)
        ]
        resampler.SetSize(out_size)
        resampler.SetOutputSpacing(desired_spacing)

        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(image.GetPixelIDValue())
        

        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            # https://simpleitk.org/doxygen/latest/html/namespaceitk_1_1simple.html#a7cb1ef8bd02c669c02ea2f9f5aa374e5
            resampler.SetInterpolator(sitk.sitkBSpline)

        resampled_image = resampler.Execute(image)
            
        return resampled_image



