def __check(config):
    import json
    import re

    if config:
        if '{' in config:
            raw    = config
            while 'range' in raw:
                nums = re.findall('(?<=range\()[0-9, ]+', raw)[0].replace(',', ' ').split()
                raw  = re.sub('range\([0-9, ]+\)', str(list(range(*map(int, nums)))), raw, 1)
            config = json.loads(raw)
        else:
            with open(config) as f:
                config = json.load(f)
    else:
        config = {}

    return config

def main(model, config):
    
    import importlib

    config = __check(config)

    config_flag   = bool(config)
    multiple_flag = any(isinstance(val, list) for val in config.values())

    
    
if __name__ == '__main__':

    from   genolearn.models import classification

    import argparse
    

    parser = argparse.ArgumentParser('genolearn.train')

    parser.add_argument('model', choices = classification.list)
    parser.add_argument('--config', default = None)

    args = parser.parse_args()

    main(args.model, args.config)

    