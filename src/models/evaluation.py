from abc import abstractmethod
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import SimpleITK as sitk

from src.models.fetch_data_from_hdf5 import get_bb_mask_voxel
from src.models.losses_2d import dice_coe_hard


class BaseEvaluator():
    def __init__(
            self,
            model,
            file,
            clinical_df,
            output_shape=(256, 256),
    ):
        self.file = file
        self.model = model
        self.clinical_df = clinical_df
        self.output_shape = output_shape

    def _compute_score(self, patient):
        pass

    def _compute_aggregated_score(self, scores):
        pass

    def _get_groundtruth(self, patient):
        pass

    def _get_plc_status(self, patient):
        patient_id = int(patient.split('_')[1])
        return float(self.clinical_df.loc[patient_id, "plc_status"])

    def _get_centered_image(self,
                            patient,
                            return_middle_slice=False,
                            return_mask=False):
        image = self.file[patient]["image"][()]
        mask = self.file[patient]["mask"][()]
        bb_lung = get_bb_mask_voxel(mask[..., 2] + mask[..., 3])
        center = ((bb_lung[:3] + bb_lung[3:]) // 2)[:2]
        r = [self.output_shape[i] // 2 for i in range(2)]
        image = image[center[0] - r[0]:center[0] + r[0],
                      center[1] - r[1]:center[1] + r[1], :, :]

        if return_mask:
            mask = mask[center[0] - r[0]:center[0] + r[0],
                        center[1] - r[1]:center[1] + r[1], :, :]

        if return_middle_slice:
            bb_gtvl = get_bb_mask_voxel(mask[..., 1])
            middle_slice = (bb_gtvl[2] + bb_gtvl[-1]) // 2
            image = image[:, :, middle_slice, :]
            image = image[np.newaxis, ...]
            if return_mask:
                mask = mask[:, :, middle_slice, :]
                mask = mask[np.newaxis, ...]
        else:
            image = np.transpose(image, (2, 0, 1, 3))
            if return_mask:
                mask = np.transpose(mask, (2, 0, 1, 3))
        if return_mask:
            return image, mask
        else:
            return image

    def __call__(self, patient_list):
        scores = list()
        for patient in patient_list:
            scores.append(self._compute_score(patient))
        return self._compute_aggregated_score(scores)


class PLCSegEvaluator(BaseEvaluator):
    def __call__(self, patient_list, path_to_save_nii=None):
        scores = list()
        for patient in patient_list:
            scores.append(
                self._compute_score(patient,
                                    path_to_save_nii=path_to_save_nii))
        return self._compute_aggregated_score(scores)

    def _compute_score(self, patient, path_to_save_nii=None):
        plc_status = self._get_plc_status(patient)
        if path_to_save_nii:
            image, mask = self._get_centered_image(patient, return_mask=True)
        else:
            image = self._get_centered_image(patient)
        prediction = self.model.predict(image, batch_size=8)
        if path_to_save_nii:
            image = np.transpose(image, (0, 2, 1, 3))
            mask = np.transpose(mask, (0, 2, 1, 3))
            prediction = np.transpose(prediction, (0, 2, 1, 3))
            ct_sitk = sitk.GetImageFromArray(image[..., 0])
            pt_sitk = sitk.GetImageFromArray(image[..., 1])
            plc_pred_sitk = sitk.GetImageFromArray(prediction[..., 1])
            gtvl_sitk = sitk.GetImageFromArray(mask[..., 1])
            sitk.WriteImage(
                ct_sitk,
                str((Path(path_to_save_nii) /
                     f"{patient}__ct.nii.gz").resolve()))
            sitk.WriteImage(
                pt_sitk,
                str((Path(path_to_save_nii) /
                     f"{patient}__pt.nii.gz").resolve()))
            sitk.WriteImage(
                plc_pred_sitk,
                str((Path(path_to_save_nii) /
                     f"{patient}__plc_pred.nii.gz").resolve()))
            sitk.WriteImage(
                gtvl_sitk,
                str((Path(path_to_save_nii) /
                     f"{patient}__gtvl.nii.gz").resolve()))

        return np.sum(prediction[..., 1] > 0.5, axis=(0, 1, 2)), plc_status

    def _compute_aggregated_score(self, scores):
        predicted_volumes = np.stack([s[0] for s in scores])
        plc_statuses = np.stack([s[1] for s in scores])
        if np.max(predicted_volumes) != 0:
            predicted_volumes = (predicted_volumes - np.min(predicted_volumes)
                                 ) / (np.max(predicted_volumes) -
                                      np.min(predicted_volumes))
        return {"AUC_seg": roc_auc_score(plc_statuses, predicted_volumes)}


class ClassificationEvaluator(BaseEvaluator):
    def _compute_score(self, patient):
        plc_status = self._get_plc_status(patient)
        image = self._get_centered_image(patient, return_middle_slice=True)
        prediction = self.model(image)
        return np.squeeze(prediction), plc_status

    def _compute_aggregated_score(self, scores):
        predicted_scores = np.stack([s[0] for s in scores])
        plc_statuses = np.stack([s[1] for s in scores])
        return {
            "AUC_classif": roc_auc_score(plc_statuses, predicted_scores[:, 1])
        }


class MultiTaskEvaluator(BaseEvaluator):
    def _compute_score(self, patient):
        plc_status = self._get_plc_status(patient)
        image = self._get_centered_image(patient)
        prediction = self.model.predict(image, batch_size=8)[0]
        pred_volume = np.sum(prediction[..., 1] > 0.5, axis=(0, 1, 2))
        image = self._get_centered_image(patient, return_middle_slice=True)
        prediction = self.model(image)
        return pred_volume, np.squeeze(prediction[1]), plc_status

    def _compute_aggregated_score(self, scores):
        predicted_volumes = np.stack([s[0] for s in scores])
        predicted_plc_statuses = np.stack([s[1] for s in scores])
        plc_statuses = np.stack([s[2] for s in scores])
        if np.max(predicted_volumes) != 0:
            predicted_volumes = (predicted_volumes - np.min(predicted_volumes)
                                 ) / (np.max(predicted_volumes) -
                                      np.min(predicted_volumes))
        return {
            "AUC_seg":
            roc_auc_score(plc_statuses, predicted_volumes),
            "AUC_classif":
            roc_auc_score(plc_statuses, predicted_plc_statuses[:, 1]),
        }
