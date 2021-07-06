from SimpleITK.SimpleITK import OrImageFilter
import tensorflow as tf
import tensorflow_addons as tfa
import scipy
import numpy as np

from src.utils.image import normalize_image, get_ct_range


@tf.function
def random_rotate(image, mask, angle=10):
    random_angle = tf.random.uniform(minval=-angle, maxval=angle,
                                     shape=(1, )) * np.pi / 180.0
    image = tfa.image.rotate(image, random_angle, fill_mode="reflect")
    mask = tfa.image.rotate(mask,
                            random_angle,
                            interpolation="nearest",
                            fill_mode="reflect")
    return image, mask


def get_bb_mask_voxel(mask):
    positions = np.where(mask != 0)
    x_min = np.min(positions[0])
    y_min = np.min(positions[1])
    z_min = np.min(positions[2])
    x_max = np.max(positions[0])
    y_max = np.max(positions[1])
    z_max = np.max(positions[2])
    return np.array([x_min, y_min, z_min, x_max, y_max, z_max])


def transform_matrix_offset(matrix, center):
    o_x, o_y, o_z = center
    offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y], [0, 0, 1, o_z],
                              [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -o_x], [0, 1, 0, -o_y], [0, 0, 1, -o_z],
                             [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(reset_matrix, matrix), offset_matrix)
    return transform_matrix


def get_rotation_matrix(angles=(0, 0, 0), center=(0, 0, 0)):
    if angles:
        theta_x = angles[0]
        theta_y = angles[1]
        theta_z = angles[2]
    else:
        theta_x = 0
        theta_y = 0
        theta_z = 0

    transform_matrix = None
    if theta_x != 0:
        theta = np.deg2rad(theta_x)
        rotation_matrix = np.array([[1, 0, 0, 0],
                                    [0, np.cos(theta), -np.sin(theta), 0],
                                    [0, np.sin(theta),
                                     np.cos(theta), 0], [0, 0, 0, 1]])
        transform_matrix = rotation_matrix

    if theta_y != 0:

        theta = np.deg2rad(theta_y)
        rotation_matrix = np.asarray([[np.cos(theta), 0,
                                       np.sin(theta), 0], [0, 1, 0, 0],
                                      [-np.sin(theta), 0,
                                       np.cos(theta), 0], [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = rotation_matrix
        else:
            transform_matrix = np.dot(transform_matrix, rotation_matrix)

    if theta_z != 0:
        theta = np.deg2rad(theta_z)
        rotation_matrix = np.asarray([[np.cos(theta), -np.sin(theta), 0, 0],
                                      [np.sin(theta),
                                       np.cos(theta), 0, 0], [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = rotation_matrix
        else:
            transform_matrix = np.dot(transform_matrix, rotation_matrix)
    if transform_matrix:
        return transform_matrix_offset(transform_matrix, center)
    else:
        return np.identity(4)


def get_shearing_matrix(shears, center=(0, 0, 0)):
    shear_xy, shear_xz, shear_yz = shears[0], shears[1], shears[2],
    transform_matrix = None
    if shear_xy != 0:
        shear = np.deg2rad(shear_xy)
        shear_matrix = np.array([[1, -np.sin(shear), 0, 0],
                                 [0, np.cos(shear), 0, 0], [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if shear_xz != 0:
        shear = np.deg2rad(shear_xz)
        shear_matrix = np.array([[1, 0, -np.sin(shear), 0], [0, 1, 0, 0],
                                 [0, 0, np.cos(shear), 0], [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if shear_yz != 0:
        shear = np.deg2rad(shear_yz)
        shear_matrix = np.array([[1, 0, 0, 0], [0, 1, -np.sin(shear), 0],
                                 [0, 0, np.cos(shear), 0], [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if transform_matrix:
        return transform_matrix_offset(transform_matrix, center)
    else:
        return np.identity(4)


def get_shifting_matrix(shifts):
    tx, ty, tz = shifts[0], shifts[1], shifts[2],
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
    ])


def get_random_shift(bb_mask,
                     image_shape,
                     output_shape=(256, 256, 1),
                     probability_include_mask=1.0):
    if len(output_shape) == 2:
        output_shape = tuple(output_shape) + (1, )
    elif len(output_shape) != 3:
        raise ValueError(
            "Fuck you, that's not a fuckcimng shape that you're feeding me !!")
    center_mask = (bb_mask[3:] + bb_mask[:3]) / 2
    max_radius = (output_shape[0] // 2)
    if np.random.random() < probability_include_mask:
        max_radius -= np.min(bb_mask[3:5] - bb_mask[:2]) / 2
    else:
        max_radius += np.max(bb_mask[3:5] - bb_mask[:2])

    radius = np.sqrt(np.random.uniform(high=max_radius**2))
    # radius = max_radius
    theta = np.random.uniform(high=2 * np.pi)
    # delta_z = np.random.normal(scale=(bb_mask[-1] - bb_mask[2]) / 2 / 1.96)

    dz = 0.9 * (bb_mask[5] - bb_mask[2]) / 2
    delta_z = np.random.uniform(low=-dz, high=dz)
    # delta_z = 0
    delta_position = np.array(
        [radius * np.cos(theta), radius * np.sin(theta), delta_z])

    center = center_mask + delta_position
    center = recenter_center(center, output_shape, image_shape)
    return center - center_mask


def recenter_center(center, output_shape, image_shape):
    pos_max = center + (np.array(output_shape) // 2)
    pos_min = center - (np.array(output_shape) // 2)
    # Check if the the output image is outside the image and correct it
    delta_out = image_shape[3:] - pos_max
    delta_out[delta_out > 0] = 0
    new_center = center + delta_out

    delta_out = image_shape[:3] - pos_min
    delta_out[delta_out < 0] = 0
    new_center = new_center + delta_out

    return new_center


def get_bb_mask_np(mask):
    positions = np.where(mask != 0)
    z_min = np.min(positions[0])
    y_min = np.min(positions[1])
    x_min = np.min(positions[2])
    z_max = np.max(positions[0])
    y_max = np.max(positions[1])
    x_max = np.max(positions[2])
    return np.array([x_min, y_min, z_min, x_max, z_max, y_max])


def get_transform_matrix(
        mat,
        output_shape=(224, 224, 1),
        spacing=(1, 1, 1),
        augment_angles=(0, 0, 0),
        random_center=True,
        mask_of_interest=None,
):

    bb_mask_vx = get_bb_mask_voxel(mask_of_interest)
    spacing_in = np.dot(mat, [1, 1, 1, 1])[:3]
    out_shape = np.array(output_shape) * np.array(spacing_in) / np.array(
        spacing)

    center_vx = (bb_mask_vx[3:] + bb_mask_vx[:3]) / 2
    center_mm = np.dot(mat, np.array([*center_vx, 1]))[:3]

    if augment_angles != (0, 0, 0):
        rotation_matrix = get_rotation_matrix(angles=augment_angles,
                                              center=center_mm)
    else:
        rotation_matrix = np.eye(4)

    if random_center:
        center_out_vx = get_random_shift(bb_mask_vx,
                                         mask_of_interest.shape,
                                         output_shape=out_shape)
        center_out_mm = np.dot(mat, np.array([*center_out_vx, 1]))[:3]
    else:
        center_out_mm = center_mm

    return rotation_matrix, center_out_mm


def slice_image(ct,
                pt,
                mask_lung1,
                mask_lung2,
                mask_gtvt,
                mask_gtvl,
                coordinate_matrix_ct,
                coordinate_matrix_pt,
                coordinate_matrix_mask_lung,
                augment_angles=(0, 0, 0),
                output_shape=(256, 256),
                spacing=(1, 1, 1),
                random_center=False,
                interp_order_image=3,
                interp_order_mask=0,
                fill_mode='constant',
                fill_value=0.0):
    mask_lung = mask_lung1 + mask_lung2
    # mask_lung[np.where(mask_lung != 0)] = 1

    if len(output_shape) == 2:
        output_shape = tuple(output_shape) + (1, )
        output_shape = np.array(output_shape)
    elif len(output_shape) != 3:
        raise ValueError(
            "Fuck you, that's not a fuckcimng shape that you're giving me !!")

    rotation_matrix_mm, center_mm = get_transform_matrix(
        coordinate_matrix_ct,
        output_shape=output_shape,
        spacing=spacing,
        augment_angles=augment_angles,
        random_center=random_center,
        mask_of_interest=mask_gtvt,
    )

    origin = center_mm - (output_shape // 2) * spacing
    coordinate_matrix_output = compute_coordinate_matrix(
        origin=origin, pixel_spacing=spacing, direction=np.eye(3).flatten())

    mat_trans = np.dot(rotation_matrix_mm, coordinate_matrix_output)
    mat_ct = np.dot(np.linalg.inv(coordinate_matrix_ct), mat_trans)
    mat_pt = np.dot(np.linalg.inv(coordinate_matrix_pt), mat_trans)
    mat_lung = np.dot(np.linalg.inv(coordinate_matrix_mask_lung), mat_trans)

    im = np.stack([
        scipy.ndimage.interpolation.affine_transform(image,
                                                     mat[:3, :3],
                                                     mat[:3, 3],
                                                     order=interp_order_image,
                                                     output_shape=output_shape,
                                                     mode=fill_mode,
                                                     cval=fill_value)
        for image, mat in [(ct, mat_ct), (pt, mat_pt)]
    ],
                  axis=-1)
    msk = np.stack(
        [
            scipy.ndimage.interpolation.affine_transform(
                mask,
                mat[:3, :3],
                mat[:3, 3],
                order=interp_order_mask,
                output_shape=output_shape,
                mode=fill_mode,
                cval=fill_value) for mask, mat in [
                    (mask_lung, mat_lung),
                    # (mask_lung2, mat_lung),
                    (mask_gtvt, mat_ct),
                    (mask_gtvl, mat_ct),
                ]
        ],
        axis=-1)
    msk[msk > 0.5] = 1
    msk[msk < 0.5] = 0,

    im = np.squeeze(im)
    im = np.stack(
        [im[..., 0],
         normalize_image(im[..., 1]),
         np.zeros_like(im[..., 1])],
        axis=-1)
    return im, np.squeeze(msk)


def compute_coordinate_matrix(
    origin=None,
    pixel_spacing=None,
    direction=None,
):
    return np.array([
        [
            direction[0] * pixel_spacing[0],
            direction[3] * pixel_spacing[1],
            direction[6] * pixel_spacing[2], origin[0]
        ],
        [
            direction[1] * pixel_spacing[0],
            direction[4] * pixel_spacing[1],
            direction[7] * pixel_spacing[2], origin[1]
        ],
        [
            direction[2] * pixel_spacing[0],
            direction[5] * pixel_spacing[1],
            direction[8] * pixel_spacing[2], origin[2]
        ],
        [0, 0, 0, 1],
    ], )  # yapf: disable
