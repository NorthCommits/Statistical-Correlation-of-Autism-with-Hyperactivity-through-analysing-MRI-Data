import yaml
import os

# Default path to the YAML file (relative to this script)
DEFAULT_YAML = os.path.join(os.path.dirname(__file__), "HyperactivityKnowledge.yaml")


def load_knowledge(yaml_path: str = None):
    """
    Load the hyperactivity traits from a YAML file.

    Args:
        yaml_path (str, optional): Path to the YAML file. If not provided,
            will use the default HyperactivityKnowledge.yaml in this directory.

    Returns:
        List[dict]: List of trait dictionaries under the `hyperactivity_traits` key.

    Raises:
        FileNotFoundError: If the YAML file cannot be found.
        KeyError: If the expected key is missing in the YAML structure.
        yaml.YAMLError: If the YAML content is invalid.
    """
    path = yaml_path or DEFAULT_YAML

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Unable to find YAML file at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    try:
        return data["hyperactivity_traits"]
    except KeyError:
        raise KeyError("Expected 'hyperactivity_traits' key not found in the YAML.")
