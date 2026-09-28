
import pathlib
import sys

def get_third_party_dir(subdir=None):
    current_dir = pathlib.Path(__file__).parent.absolute()
    third_party_dir = current_dir.parent / 'third_party'
    if subdir is not None:
        third_party_dir = third_party_dir / subdir
    return third_party_dir

def add_package_path(package_name=None):
    third_party_dir = get_third_party_dir(package_name)
    sys.path.append(str(third_party_dir))