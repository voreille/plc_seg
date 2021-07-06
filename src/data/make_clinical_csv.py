import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import h5py

project_dir = Path(__file__).resolve().parents[2]

path_clinical_info_original = project_dir / "data/List_lymphangite_radiomics_SNM2020_MJ.csv"
hdf5_filepath = project_dir / "data/processed/2d_pet_normalized/data.hdf5"
output_path = project_dir / "data/clinical_info.csv"


def main():
    df = pd.read_csv(path_clinical_info_original).set_index("patient_id")
    out_df = pd.DataFrame()
    with h5py.File(hdf5_filepath, "r") as f:
        for p in f.keys():
            patient_id = int(p.split("_")[1])
            mask = f[p]["mask"][()]
            gtvlt = mask[:, :, :, 0] + mask[:, :, :, 1]
            if np.sum(gtvlt * mask[:, :, :, 2]) != 0:
                a = 2
            elif np.sum(gtvlt * mask[:, :, :, 3]) != 0:
                a = 3
            else:
                warnings.warn(f"mec le patient {p} il a un prob")
                a = np.nan
            out_df = out_df.append(
                {
                    "patient_id": patient_id,
                    "sick_lung_axis": a,
                },
                ignore_index=True,
            )
    out_df["patient_id"] = out_df["patient_id"].astype(int)
    out_df = out_df.set_index("patient_id")
    out_df = pd.concat([df, out_df], axis=1)
    out_df.to_csv(output_path)


if __name__ == '__main__':
    # project_dir = Path(__file__).resolve().parents[2]
    main()
