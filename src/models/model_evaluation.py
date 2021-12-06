import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.models.fetch_data_from_hdf5 import get_bb_mask_voxel


def compute_plc_volume(model,
                       file,
                       clinical_df,
                       image_shape=(224, 224, 3),
                       batch_size=4,
                       threshold=0.5):
    df = pd.DataFrame()
    patient_list = list(file.keys())
    patient_list = [p for p in patient_list if p not in ["PatientLC_63"]]
    print("Computing PLC volume")
    for p in tqdm(patient_list):
        image = file[p]["image"][()]
        mask = file[p]["mask"][()]

        patient_id = int(p.split('_')[1])
        sick_lung_axis = int(clinical_df.loc[int(patient_id),
                                             "sick_lung_axis"])
        bb_gtvl = get_bb_mask_voxel(mask[..., 1])
        bb_gtvt = get_bb_mask_voxel(mask[..., 0])
        z_min = np.min([bb_gtvl[2], bb_gtvt[2]])
        z_max = np.max([bb_gtvl[5], bb_gtvt[5]])
        bb_lung = get_bb_mask_voxel(mask[..., 2] + mask[..., 3])
        center = ((bb_lung[:3] + bb_lung[3:]) // 2)[:2]
        r = [image_shape[i] // 2 for i in range(2)]
        image = image[center[0] - r[0]:center[0] + r[0],
                      center[1] - r[1]:center[1] + r[1],
                      z_min:z_max + 1, :image_shape[2]]
        mask = mask[center[0] - r[0]:center[0] + r[0],
                    center[1] - r[1]:center[1] + r[1], z_min:z_max + 1, :]
        image = np.transpose(image, (2, 0, 1, 3))
        mask = np.transpose(mask, (2, 0, 1, 3))
        y_pred = model.predict(image, batch_size=batch_size)
        if sick_lung_axis == 2:
            contro_axis = 3
        elif sick_lung_axis == 3:
            contro_axis = 2
        lung_sick = mask[..., sick_lung_axis]
        lung_contro = mask[..., contro_axis]
        df = df.append(
            {
                "patient_name":
                p,
                "patient_id":
                patient_id,
                "plc_volume":
                np.sum(y_pred[..., 1] > threshold),
                "plc_volume_sick":
                np.sum(y_pred[..., 1] * lung_sick > threshold),
                "plc_volume_controlateral":
                np.sum(y_pred[..., 1] * lung_contro > threshold),
            },
            ignore_index=True)
    return df.set_index("patient_id")


def compute_results(volume_df, clinical_df):
    df = pd.concat([clinical_df, volume_df], axis=1)
    df = df.dropna(axis=0)
    score_train, score_train_lung = compute_score(
        df[df["train_val_test"] == 0])
    score_val, score_val_lung = compute_score(df[df["train_val_test"] == 1])
    score_test, score_test_lung = compute_score(df[df["train_val_test"] == 2])
    return {
        "auc_train": score_train,
        "auc_train_lung": score_train_lung,
        "auc_val": score_val,
        "auc_val_lung": score_val_lung,
        "auc_test": score_test,
        "auc_test_lung": score_test_lung,
    }


def compute_score(input_df):
    df = input_df.copy()
    df.loc[:, "plc_volume_standardized"] = df.loc[:, "plc_volume"] / np.max(
        df["plc_volume"])
    df["plc_volume_sick_standardized"] = df["plc_volume_sick"].map(
        lambda x: x / np.max([
            np.max(df["plc_volume_sick"]),
            np.max(df["plc_volume_controlateral"])
        ]))
    df["plc_volume_controlateral_standardized"] = df[
        "plc_volume_controlateral"].map(lambda x: x / np.max([
            np.max(df["plc_volume_sick"]),
            np.max(df["plc_volume_controlateral"])
        ]))
    X = np.concatenate(
        [
            df["plc_volume_sick_standardized"].values,
            df["plc_volume_controlateral_standardized"].values
        ],
        axis=0,
    )
    y = np.concatenate(
        [df["plc_status"].values,
         np.zeros(df["plc_status"].values.shape)],
        axis=0,
    )
    score = roc_auc_score(df["plc_status"], df["plc_volume_standardized"])
    score_lung = roc_auc_score(y, X)
    return score, score_lung
