import argparse
import importlib
import sys


def main(args):
    module_path = args[0]
    module = importlib.import_module(module_path)
    module.main(args[1:])

if __name__ == '__main__':
    main(sys.argv[1:])



