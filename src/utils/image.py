import numpy as np
import SimpleITK as sitk


def to_np(x):
    return np.squeeze(np.transpose(sitk.GetArrayFromImage(x), (2, 1, 0)))


def get_bb_image_mm(image):
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
    return np.array([x_min, y_min, z_min, x_max, z_max, y_max])


def get_bb_mask_mm(mask_sitk):
    bb = get_bb_mask_voxel(mask_sitk)
    return np.array([
        *mask_sitk.TransformIndexToPhysicalPoint(
            [int(bb[0]), int(bb[1]), int(bb[2])]),
        *mask_sitk.TransformIndexToPhysicalPoint(
            [int(bb[3]), int(bb[4]), int(bb[5])])
    ])


def get_ct_range(anat_str):
    wwl_dict = {
        "lung": [1500, -600],
        "brain": [80, 40],
        "head_neck_soft_tissue": [350, 60],
    }

    return (wwl_dict[anat_str][1] - wwl_dict[anat_str][0] * 0.5,
            wwl_dict[anat_str][1] + wwl_dict[anat_str][0] * 0.5)


def normalize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std
