import yaml

class DotDict:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return f"{self.__dict__}"

def load_config(path):
    with open(path, 'r') as f:
        config=yaml.safe_load(f)
        args = DotDict()
        for key, value in config.items():
            args[key] = value
        return args

def main(args):
    # Load configuration
    args = load_config(args.config)

    print(args.model)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test script for processing images.")
    parser.add_argument('--config', type=str, help='First argument for testing')

    args = parser.parse_args()
    main(args)