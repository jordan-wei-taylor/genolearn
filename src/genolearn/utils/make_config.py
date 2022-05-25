if __name__ == '__main__':
    from   genolearn.utils import check_config
    import argparse
    import json

    parser = argparse.ArgumentParser(description = 'make_config.py\n\nGenerates a config file from string')

    parser.add_argument('outpath')
    parser.add_argument('config')

    args = parser.parse_args()

    config = check_config(args.config)

    with open(args.outpath, 'w') as f:
        f.write(json.dumps(config, indent = 4))
