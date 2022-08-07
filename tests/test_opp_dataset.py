import pytest
import os
from src.opp_dataset import OppDataset
import requests


# @pytest.fixture(scope="session")
# def data_dir(tmp_path_factory):
#     tmp_data_dir = tmp_path_factory.mktemp("tmp_data_dir")
#     OppDataset.download_opportunity(to=tmp_data_dir)
#     return tmp_data_dir

@pytest.fixture(scope="session")
def opp_data():
    return OppDataset(dataset_dir="./data/")


def test_nonexistent_data_dir():
    with pytest.raises(FileNotFoundError):
        OppDataset(dataset_dir="nonexistent_dir")

# def test_init_opp_data(data_dir):
#     opp_pp = OppDataset(dataset_dir=data_dir)
#     print(opp_pp.get_col_list())
#     assert len(opp_pp.get_col_list()) == 250


def test_get_col_list(opp_data):
    assert len(opp_data.get_col_list()) == 250




