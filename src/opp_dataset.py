from typing import List, Tuple, Union
import requests
from tqdm import tqdm
import os
from io import BytesIO
import errno
from collections import Counter
import re
from zipfile import ZipFile
import warnings
import pandas as pd
from pathlib import Path, PosixPath

import _utils


class OppDataset:
    # Consts
    SENSOR_SAMPLE_RATE = 30  # 30 Hz
    SUBJECTS: Tuple[str] = ("S1", "S2", "S3", "S4")
    RUNS: Tuple[str] = ("ADL1", "ADL2", "ADL3", "ADL4", "ADL5", "Drill")

    ACC_SENSORS: Tuple[str] = ("RKN^", "RKN_", "RUA_", "RUA^", "RWR", "RH", "LUA^", "LH", "LUA_", "LWR", "HIP", "BACK")
    IMU_SENSORS: Tuple[str] = ("BACK", "RUA", "RLA", "LUA", "LLA", "L-SHOE", "R-SHOE")
    LABEL_COLUMNS: List[str] = ["Locomotion", "HL_Activity", "LL_Left_Arm", "LL_Left_Arm_Object", "LL_Right_Arm",
                                "LL_Right_Arm_Object", "ML_Both_Arms"]
    SENSOR_TYPES: Tuple[str] = ("acc", "gyro", "magnetic", "Quaternion")
    ZIP_FILE_NAME: str = "OpportunityUCIDataset.zip"
    DEFAULT_DATASET_DIR: str = "./data/"
    ARCHIVE_URL: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip"

    dataset_path = None

    @staticmethod
    def download_opportunity(to: Union[Path, str] = "./data/", archive_url: str = ARCHIVE_URL):
        """
        Download the OPPORTUNITY dataset to the given path.

        :param to: directory the dataset zip file is to be saved
        :param archive_url:
        :return:
        """

        if type(to) != PosixPath:
            to = Path(to)

        if not to.is_dir():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT),
                f"{to} not found. Please specify a directory that already exists."
            )

        res = requests.get(archive_url, stream=True)
        chunk_size = 1024
        with open(to / "OpportunityUCIDataset.zip", "wb") as f:
            total_length = int(res.headers.get('content-length'))
            for chunk in tqdm(res.iter_content(chunk_size=chunk_size), total=int(total_length / chunk_size) + 1,
                              desc="Downloading OpportunityUCIDataset.zip..."):
                if chunk:
                    f.write(chunk)
                    f.flush()

            print(f"OpportunityUCIDataset.zip has been saved in {to}")

        return res

    def __init__(self, dataset_dir: str = DEFAULT_DATASET_DIR, download: bool = False):
        """

        :param dataset_dir:
        """

        # Check if the directory exists.
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT),
                f"{dataset_dir} {'(Default directory)' if dataset_dir == self.DEFAULT_DATASET_DIR else ''} not found. "
                f"Please specify a directory that already exists."
            )

        dataset_path = dataset_dir + f"{'' if dataset_dir[-1] == '/' else '/'}" + self.ZIP_FILE_NAME

        # Check if the dataset file exists.
        if not os.path.isfile(dataset_path):
            # Download the dataset when download is True
            if download:
                # Check if dataset_dir exists
                self.download_opportunity(to=dataset_dir)
            else:
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT),
                    f"{dataset_path} (default path) not found. Specify `dataset_dir` in the constructor that contains "
                    f"{self.ZIP_FILE_NAME} or download the dataset file in `dataset_dir` by passing `download=True` "
                    f"to the constructor. The default `dataset_dir` is '{self.DEFAULT_DATASET_DIR}'")

        self.dataset_path = dataset_path

    def get_col_list(self) -> List[str]:
        """
        Return the column names of the Opportunity dataset
        Parameters
        column_names_txt_path (optional): the default path is self.dataset_path+"column_names.txt"
        ----------
        Returns list of columns of the Opportunity dataset
        -------
        """

        # Check if the dataset exists at self.dataset_path
        self._validate_filepath()

        opp_sensor_columns = []
        regx_str = r"Column: (?P<col_no>\d+) (?P<col_name>[\w\s^-]+)(;\s(?P<meta_info>[\w\s^=(),/*]+))*\n*"
        with ZipFile(OppDataset.ZIP_FILE_NAME, 'r') as opp_zip:
            for line in opp_zip.read("OpportunityUCIDataset/dataset/column_names.txt").decode('utf-8').split("\r\n"):
                match = re.search(
                    regx_str,
                    line,
                )
                if match:
                    opp_sensor_columns.append(match.group("col_name").replace("\n", ""))
                else:
                    # Only rows with these strings should be ignored
                    assert line in ["Data columns:", "Label columns: ", ""], line

            # Alter duplicated column names
            dup_items = dict(filter(lambda item: item[1] > 1, Counter(opp_sensor_columns).items()))
            for dup_item in dup_items.keys():
                dup_item_idx = opp_sensor_columns.index(dup_item)
                opp_sensor_columns[dup_item_idx + 1] = dup_item[:-1] + "Y"
                opp_sensor_columns[dup_item_idx + 2] = dup_item[:-1] + "Z"

            # Check if there is any duplicated columns left
            assert (dict(filter(lambda item: item[1] > 1, Counter(opp_sensor_columns).items())) == {})

        return opp_sensor_columns

    def get_filepath_of_sbj_run(self, subject: str, run: str) -> str:
        """
        Return the filepath of the specified run of the specified subject.
        :param subject: "S1", "S2", "S3", or "S4"
        :param run: "ADL1", "ADL2", "ADL3", "ADL4", "ADL5", or "Drill"
        :return:
        """

        # parameter validation
        self._validate_sbj_run(subject, run)
        file_path = f"OpportunityUCIDataset/dataset/{subject}-{run}.dat"

        # Check if file exists in zip.
        assert file_path in _utils.get_file_list_in_zip(self.dataset_path)

        return file_path

    def load_file(self, subject: str, run: str, fillna: bool = False, set_timestamp_as_index: bool = False) \
            -> pd.DataFrame:
        """
        Load a sensor data file of the specified run of the specified subject and return the data in the DataFrame
        format.
        :param subject: target subject "S1", "S2", "S3", or "S4"
        :param run: "ADL1", "ADL2", "ADL3", "ADL4", "ADL5", or "Drill"
        :param fillna:
        :param set_timestamp_as_index:
        :return: data
        """

        # parameter validation
        self._validate_sbj_run(subject, run)

        # prepare file_path and opp_columns
        file_path = self.get_filepath_of_sbj_run(subject, run)
        opp_columns = self.get_col_list()

        # load .dat file and covert it to dataframe
        with ZipFile(self.dataset_path, 'r') as opp_zip:
            b_data = opp_zip.read(file_path)
            df_data = pd.read_csv(BytesIO(b_data), sep=' ', header=None, names=opp_columns)

        if fillna:
            df_data = _utils.replace_nans(df_data)
            if df_data.isnull().values.any():
                warnings.warn("nan values still remain in the data")

        if set_timestamp_as_index:
            df_data = df_data.set_index("MILLISEC")

        return df_data

    def get_acc_columns(self, acc_sensors: List[str]) -> List[str]:
        """ filter columns of the opportunity dataset by list of sensor names and types
        :param acc_sensors:
        :return: list of filtered columns
        """

        assert set(acc_sensors).issubset(self.ACC_SENSORS),\
            f"No columns of {set(acc_sensors).difference(self.ACC_SENSORS)} Accelerometer sensor(s) exist."

        opp_cols = self.get_col_list()
        return [col for col in opp_cols if col.split()[0] == "Accelerometer" and col.split()[1] in acc_sensors]

    def get_imu_columns(self, imu_sensors: List[str], sensor_types: List[str]) -> List[str]:
        """ filter columns of the opportunity dataset by a list of sensor names and a list of sensor types
        :param imu_sensors: subset of OppDataset.IMU_SENSORS
        :param sensor_types: subset of OppDataset.SENSOR_TYPES
        :return: list of filtered columns
        """

        assert set(imu_sensors).issubset(self.IMU_SENSORS),\
            f"No columns of {set(imu_sensors).difference(self.IMU_SENSORS)} IMU sensor(s) exist."
        assert set(sensor_types).issubset(self.SENSOR_TYPES),\
            f"Sensor type(s) {set(sensor_types).difference(self.SENSOR_TYPES)} do(es) not exist."

        opp_cols = self.get_col_list()
        return [col for col in opp_cols if col.split()[0] == "InertialMeasurementUnit" in col
                and col.split()[1] in imu_sensors and col.split()[-1][:-1] in sensor_types]

    # Private
    def _validate_filepath(self):
        assert os.path.isfile(self.dataset_path), f"{self.dataset_path} not found"

    def _validate_sbj_run(self, subject: str, run: str) -> None:
        """
        Validate if the given subject and run parameters are valid
        :param subject: "S1", "S2", "S3", or "S4"
        :param run: "ADL1", "ADL2", "ADL3", "ADL4", "ADL5", "Drill"
        :return: None
        """
        assert subject in self.SUBJECTS, f"`subject` must be one of {self.SUBJECTS}"
        assert run in self.RUNS, f"`run` must be one of {self.RUNS}"


