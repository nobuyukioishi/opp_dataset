import pytest
import os
from src.opp_dataset import OppDataset
import requests

tmp_url = "https://i.picsum.photos/id/955/200/200.jpg?hmac=_m3ln1pswsR9s9hWuWrwY_O6N4wizKmukfhvyaTrkjE"

@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    tmp_data_dir = tmp_path_factory.mktemp("data")
    return tmp_data_dir

@pytest.fixture(scope="session")
def hoge_image(data_dir):
    r = requests.get(tmp_url)
    img = data_dir / "hoge.img"
    img.write_bytes(r.content)
    # OppDataset.download_opportunity(to=str(tmp_data_dir))
    opp_path = data_dir / OppDataset.ZIP_FILE_NAME
    return img

@pytest.fixture(scope="session")
def opp_path(data_dir):
    # OppDataset.download_opportunity(to=data_dir)
    r = requests.get(tmp_url)
    with open(data_dir / "OpportunityUCIDataset.zip", "wb") as f:
        f.write(r.content)
    return data_dir / "OpportunityUCIDataset.zip"

def test_file_exists(opp_path):
    assert os.path.isfile(opp_path), [path for path in data_dir.glob("**")]


def test_nonexistent_data_dir():
    with pytest.raises(FileNotFoundError):
        OppDataset(dataset_dir="nonexistent_dir")

