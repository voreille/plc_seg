import numpy as np
import SimpleITK as sitk

from src.utils.image import get_ct_range, normalize_image


def to_np(x):
    return np.squeeze(np.transpose(sitk.GetArrayFromImage(x), (2, 1, 0)))


def get_bb_image(image):
    origin = image.GetOrigin()
    max_position = origin + np.array(
        [image.GetWidth(),
         image.GetHeight(),
         image.GetDepth()]) * image.GetSpacing()
    return np.concatenate([origin, max_position], axis=0)


def intersect_bb(*args):
    pos_max = args[0][3:]
    pos_min = args[0][:3]
    for bb in args:
        pos_max = np.minimum(pos_max, bb[3:])
        pos_min = np.maximum(pos_min, bb[:3])

    return np.concatenate([pos_min, pos_max], axis=0)


def get_bb_mask_voxel(mask_sitk):
    mask = sitk.GetArrayFromImage(mask_sitk)
    positions = np.where(mask != 0)
    z_min = np.min(positions[0])
    y_min = np.min(positions[1])
    x_min = np.min(positions[2])
    z_max = np.max(positions[0])
    y_max = np.max(positions[1])
    x_max = np.max(positions[2])
    return x_min, y_min, z_min, x_max, z_max, y_max


def get_bb_mask_mm(mask_sitk):
    x_min, y_min, z_min, x_max, z_max, y_max = get_bb_mask_voxel(mask_sitk)
    return (*mask_sitk.TransformIndexToPhysicalPoint(
        [int(x_min), int(y_min), int(z_min)]),
            *mask_sitk.TransformIndexToPhysicalPoint(
                [int(x_max), int(y_max), int(z_max)]))


def split_lung_mask(lung_sitk):
    lung = sitk.GetArrayFromImage(lung_sitk)
    lung1 = np.zeros_like(lung, dtype=int)
    lung2 = np.zeros_like(lung, dtype=int)
    lung1[lung == 1] = 1
    lung2[lung == 2] = 1
    lung1 = sitk.GetImageFromArray(lung1)
    lung1.SetOrigin(lung_sitk.GetOrigin())
    lung1.SetSpacing(lung_sitk.GetSpacing())
    lung1.SetDirection(lung_sitk.GetDirection())
    lung2 = sitk.GetImageFromArray(lung2)
    lung2.SetOrigin(lung_sitk.GetOrigin())
    lung2.SetSpacing(lung_sitk.GetSpacing())
    lung2.SetDirection(lung_sitk.GetDirection())
    return lung1, lung2


def parse_image(
    patient,
    path_nii,
    path_lung_mask_nii,
    spacing=(1, 1, 1),
    padding=10,
    minimal_output_size=(256, 256, 1),
    interp_order=3,
    smoothing_radius=None,
    ct_window_str="lung",
):
    """Parse the raw data of HECKTOR 2020

    Args:
        folder_name ([Path]): the path of the folder containing 
        the 3 sitk images (ct, pt and mask)
    """
    # patient_name = patient.numpy().decode("utf-8")
    patient_name = patient
    ct_sitk = sitk.ReadImage(
        str((path_nii / (patient_name + "__CT.nii.gz")).resolve()))
    pt_sitk = sitk.ReadImage(
        str((path_nii / (patient_name + "__PT.nii.gz")).resolve()))
    mask_gtvt_sitk = sitk.ReadImage(
        str((path_nii /
             (patient_name + "__GTV_T__RTSTRUCT__CT.nii.gz")).resolve()))
    mask_gtvl_sitk = sitk.ReadImage(
        str((path_nii /
             (patient_name + "__GTV_L__RTSTRUCT__CT.nii.gz")).resolve()))
    mask_lung_sitk = sitk.ReadImage(
        str((path_lung_mask_nii /
             (patient_name + "__LUNG__SEG__CT.nii.gz")).resolve()))
    resampler = sitk.ResampleImageFilter()
    if interp_order == 3:
        resampler.SetInterpolator(sitk.sitkBSpline)

    # bb_max = intersect_bb(get_bb_image(ct_sitk), get_bb_image(pt_sitk))
    # bb_lung = get_bb_mask_mm(mask_lung_sitk)
    # bb_max[2] = bb_lung[2] - padding  # crop on the z axis
    # bb_max[5] = bb_lung[5] + padding
    bb_max = get_bb_mask_mm(mask_lung_sitk)
    bb_max[:3] = bb_max[:3] - padding  # crop on the z axis
    bb_max[3:] = bb_max[3:] + padding
    if bb_max[3] - bb_max[0] < minimal_output_size[0] * spacing[0]:
        bb_max[3] = minimal_output_size[0] * spacing[0] / 2 + (bb_max[3] +
                                                               bb_max[0]) / 2
        bb_max[0] = -minimal_output_size[0] * spacing[0] / 2 + (bb_max[3] +
                                                                bb_max[0]) / 2

    if bb_max[4] - bb_max[1] < minimal_output_size[1] * spacing[1]:
        bb_max[4] = minimal_output_size[1] * spacing[1] / 2 + (bb_max[4] +
                                                               bb_max[1]) / 2
        bb_max[1] = -minimal_output_size[1] * spacing[1] / 2 + (bb_max[4] +
                                                                bb_max[1]) / 2

    if bb_max[5] - bb_max[2] < minimal_output_size[2] * spacing[2]:
        bb_max[5] = minimal_output_size[2] * spacing[2] / 2 + (bb_max[5] +
                                                               bb_max[2]) / 2
        bb_max[2] = -minimal_output_size[2] * spacing[2] / 2 + (bb_max[5] +
                                                                bb_max[2]) / 2

    output_shape = np.round((bb_max[3:] - bb_max[:3]) / spacing)
    resampler.SetOutputOrigin(bb_max[3:])
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(tuple([int(k) for k in output_shape]))

    ct_sitk = resampler.Execute(ct_sitk)
    pt_sitk = resampler.Execute(pt_sitk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_gtvt_sitk = resampler.Execute(mask_gtvt_sitk)
    mask_gtvl_sitk = resampler.Execute(mask_gtvl_sitk)
    mask_lung_sitk = resampler.Execute(mask_lung_sitk)
    mask_lung1_sitk, mask_lung2_sitk = split_lung_mask(mask_lung_sitk)
    if smoothing_radius:
        smoother = sitk.BinaryMedianImageFilter()
        smoother.SetRadius(int(smoothing_radius))
        mask_gtvt_sitk = smoother.Execute(mask_gtvt_sitk)
        mask_gtvl_sitk = smoother.Execute(mask_gtvl_sitk)
        mask_lung1_sitk = smoother.Execute(mask_lung1_sitk)
        mask_lung2_sitk = smoother.Execute(mask_lung2_sitk)
    mask_gtvt = to_np(mask_gtvt_sitk)
    mask_gtvl = to_np(mask_gtvl_sitk)
    mask_lung1 = to_np(mask_lung1_sitk)
    mask_lung2 = to_np(mask_lung2_sitk)

    ct = to_np(ct_sitk)
    pt = to_np(pt_sitk)

    hu_low, hu_high = get_ct_range(ct_window_str)
    ct[ct > hu_high] = hu_high
    ct[ct < hu_low] = hu_low
    ct = (2 * ct - hu_high - hu_low) / (hu_high - hu_low)

    pt = normalize_image(pt)

    image = np.stack([ct, pt, np.zeros_like(ct)], axis=-1)
    mask = np.stack([mask_gtvt, mask_gtvl, mask_lung1, mask_lung2], axis=-1)
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0

    return image, mask
