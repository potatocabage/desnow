import yaml


def open_configs():
    with open("configs.yml", "r") as file:
        configs = yaml.safe_load(file)

    data_configs = configs["data_configs"]
    training_configs = configs["training_configs"]
    dilation_range = configs["dilation_range"]

    return data_configs, training_configs, dilation_range