import os
import yaml

def get_yaml(name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "cifar.yaml")

    with open(config_path, 'r') as file:
        configs = list(yaml.safe_load_all(file))

    default_config = configs[0]
    for config in configs:
        if config.get('params').get('model') == name:
            return config