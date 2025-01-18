import json
from importlib import resources


def load_paths_and_configs():
    """loads all the configuration files in the config folder which are needed for the
    ZDRC calculation.

    Returns:
        dict: paths
        dict: shared_dicts
        dict: ZDRC configurations
    """
    # Access the paths.json file within the `config` directory.
    with resources.files("ZDRC_CALCULATOR_QJ_2025.config").joinpath("paths.json").open("r") as f:
        config = json.load(f)

    # Resolve relative paths to absolute paths based on the package directory.
    base_path = resources.files("ZDRC_CALCULATOR_QJ_2025")
    paths = {key: str(base_path.joinpath(value)) for key, value in config.items()}

    # Access the shared dictionaries
    with resources.open_text(
        "ZDRC_CALCULATOR_QJ_2025.config", "shared_dictionaries.json"
    ) as f:
        shared_dicts = json.load(f)

    # Access the ZDRC_algorithm configuration
    with resources.open_text(
        "ZDRC_CALCULATOR_QJ_2025.config", "zdrc_algorithm_config.json"
    ) as f:
        zdrc_algorithm_config = json.load(f)  # contains the zdr algorithm related configurations

    return paths, shared_dicts, zdrc_algorithm_config
