import os
from pathlib import Path
from random import shuffle

import dotenv
import h5py
import pandas as pd

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"

dotenv.load_dotenv(str(dotenv_path))

path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])
output_path = project_dir / "data/split.csv"


def main():
    h5_file = h5py.File(
        "/home/val/python_wkspce/plc_seg/data/processed/hdf5_2d/dataset.hdf5",
        "r")
    df = pd.read_csv(path_clinical_info).set_index("patient_id")
    patient_list = list(h5_file.keys())
    patient_list = [
        p for p in patient_list if p not in ["PatientLC_63", "PatientLC_72"]
    ]
    id_list = [int(p.split('_')[1]) for p in patient_list]
    df = df.loc[id_list, :]
    id_neg = list(df[(df["plc_status"] == 0) & (df["is_chuv"] == 1)].index)
    id_pos = list(df[(df["plc_status"] == 1) & (df["is_chuv"] == 1)].index)

    shuffle(id_neg)
    shuffle(id_pos)

    id_test_neg = id_neg[:5]
    id_test_pos = id_pos[:5]
    id_test = id_test_neg + id_test_pos

    id_val_neg = id_neg[5:8]
    id_val_pos = id_pos[5:10]
    id_val = id_val_neg + id_val_pos

    id_train = [i for i in id_list if i not in id_val + id_test]
    df.loc[id_train, "train_val_test"] = 0
    df.loc[id_val, "train_val_test"] = 1
    df.loc[id_test, "train_val_test"] = 2
    df.to_csv(output_path)


if __name__ == '__main__':
    # project_dir = Path(__file__).resolve().parents[2]
    main()