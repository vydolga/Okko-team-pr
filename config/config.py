import os
from dynaconf import Dynaconf

current_directory = os.path.dirname(os.path.realpath(__file__))

settings = Dynaconf(
    settings_files=
        [f"{current_directory}/user_features.toml",
         f"{current_directory}/item_features.toml",
         f"{current_directory}/link.toml"]
    
)
