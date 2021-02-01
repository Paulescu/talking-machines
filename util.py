def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import uuid
def get_random_id():
    return str(uuid.uuid1())

from pathlib import Path
from typing import List, Union
import tarfile
def download_artifacts(name: str, files: List[Union[str, Path]]):
    """Compresses 'files' into a 'name.tar.gz' and downloads it"""
    
    # create tar file
    with tarfile.open(f"{name}.tar.gz", "w:gz") as tar:
        for file in files:
            tar.add(file)

    # download tar file
    from google.colab import files
    files.download(f"{name}.tar.gz")
    print(f'Downloaded {name}.tar.gz')

def minutes_seconds_elapsed(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
