import warnings

import pandas as pd
import numpy as np
import h5py

path_excel = "/home/val/python_wkspce/plc_seg/data/List_lymphangite_radiomics_SNM2020_MJ.xlsx"
files = [
    "/home/val/python_wkspce/plc_seg/data/processed/2d/train.hdf5",
    "/home/val/python_wkspce/plc_seg/data/processed/2d/test.hdf5"
]
output_path = "/home/val/python_wkspce/plc_seg/data/clinical_info.csv"


def main():
    df = pd.read_excel(path_excel).set_index("patient_id")
    out_df = pd.DataFrame()
    for file in files:
        with h5py.File(file, "r") as f:
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
