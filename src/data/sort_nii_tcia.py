from pathlib import Path
import shutil

from tqdm import tqdm
import SimpleITK as sitk

project_dir = Path(__file__).resolve().parents[2]
input_filepath = project_dir / "data/interim/radiogenomics"
output_filepath = project_dir / "data/interim/radiogenomics_sorted"


def main():
    for file in tqdm(input_filepath.rglob("*PT*")):
        pt_image = sitk.ReadImage(str(file.resolve()))
        slice_thickness = pt_image.GetSpacing()[2]
        ct_files = [f for f in file.parent.rglob("*CT*")]


if __name__ == '__main__':
    # project_dir = Path(__file__).resolve().parents[2]
    main()
