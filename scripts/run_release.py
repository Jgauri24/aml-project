"""Freeze, reproduce development, then score the reserved holdout exactly once."""
import argparse
from src.release.evaluate import freeze, evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['freeze', 'development', 'final'])
    args = parser.parse_args()
    if args.operation == 'freeze':
        print(freeze())
    else:
        evaluate(args.operation)
