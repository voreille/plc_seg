import os
from pathlib import Path

import dotenv
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"
dotenv.load_dotenv(str(dotenv_path))

path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])
# model = "model_1-00_alpha_1-00_wplc_1-00_wt_1-00_wl_40-00_splc_early_stop"
# model = "model_0-75_alpha_1-00_wplc_1-00_wt_1-00_wl_40-00_splc_early_stop"
model = "model_gtvl_gtvt"
# model = "model_0-75_alpha_1-00_wplc_0-00_wt_0-00_wl_40-00_splc_early_stop"

path_volume_csv = project_dir / f"data/plc_volume/{model}.csv"


def main():
    volume_df = pd.read_csv(path_volume_csv).set_index("patient_id")
    clinical_df = pd.read_csv(path_clinical_info).set_index("patient_id")
    df = pd.concat([clinical_df, volume_df], axis=1)
    df = df.dropna(axis=0)
    df["plc_volume_standardized"] = df["plc_volume"].map(
        lambda x: x / np.max(df["plc_volume"]))
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

    print(f"that's the score {score}")
    print(f"that's the score for the lung {score_lung}")


if __name__ == '__main__':
    main()