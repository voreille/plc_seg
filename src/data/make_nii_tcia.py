from pathlib import Path

from tqdm import tqdm
import pydicom as pdcm
from pydicom.errors import InvalidDicomError

from okapy.dicomconverter.dicom_walker import DicomWalker, DicomFile
from okapy.dicomconverter.study import Study
from okapy.dicomconverter.dicom_header import DicomHeader
from okapy.dicomconverter.converter import Converter

project_dir = Path(__file__).resolve().parents[2]
dicom_path = Path(
    "/home/val/python_wkspce/plc_seg/data/raw/NSCLC_Radiogenomics/NSCLC_Radiogenomics"
)
output_filepath = project_dir / "data/interim/radiogenomics"


class TciaDicomWalker(DicomWalker):
    def __init__(
        self,
        input_dirpath=None,
    ):
        self.input_dirpath = input_dirpath

    def _walk(self, input_dirpath):
        '''
        Method to walk through the path given and fill the list of DICOM
        headers and sort them
        '''
        dicom_files = list()
        files = [f for f in Path(input_dirpath).rglob("*") if f.is_file()]
        # to test wether a slice appear multiple times
        for file in tqdm(files, desc="Walkin through all the files"):
            try:
                data = pdcm.filereader.dcmread(str(file.resolve()),
                                               stop_before_pixels=True)

            except InvalidDicomError:
                print('This file {} is not recognised as DICOM'.format(
                    file.name))
                continue
            try:
                modality = data.Modality
            except AttributeError:
                print('not reading the DICOMDIR')
                continue

            dicom_header = DicomHeader(
                patient_id=data.get("PatientID", -1),
                study_instance_uid=data.get("StudyInstanceUID", -1),
                study_date=data.get("StudyDate", -1),
                series_instance_uid=data.get("SeriesInstanceUID", -1),
                series_number=data.get("SeriesNumber", -1),
                instance_number=data.get("InstanceNumber", -1),
                image_type=data.get("ImageType", ["-1"]),
                modality=modality)
            dicom_files.append(
                DicomFile(dicom_header=dicom_header, path=str(file.resolve())))

        dicom_files.sort(key=lambda x: (
            x.dicom_header.study_instance_uid, x.dicom_header.modality, x.
            dicom_header.series_instance_uid, x.dicom_header.instance_number, x
            .dicom_header.patient_id))
        return dicom_files


def main():

    output_filepath.mkdir(parents=True, exist_ok=True)
    converter = Converter(
        str(output_filepath.resolve()),
        padding="whole_image",
        resampling_spacing=-1,
        dicom_walker=TciaDicomWalker(),
    )
    # out_folder = output_filepath / "test"
    # out_folder.mkdir(exist_ok=True)
    # result = converter(str(dicom_path.resolve()), output_folder=out_folder)

    for folder in dicom_path.iterdir():
        out_folder = output_filepath / folder.name
        out_folder.mkdir(exist_ok=True)
        try:
            result = converter(str(folder.resolve()), output_folder=out_folder)
        except Exception as e:
            print(e)
            continue

    print(result)


if __name__ == '__main__':
    # project_dir = Path(__file__).resolve().parents[2]
    main()
