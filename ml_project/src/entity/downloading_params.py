from dataclasses import dataclass, field

@dataclass()
class DownloadingParams:
    """ Structure for train model parameters """
    file_link: str = field()
    output_folder: str = field(default="data/raw/")
    name: str = field(default="data.zip")
