import argparse


def args2dict(namespace):
    is_args: lambda x: isinstance(x, argparse.Namespace)
    return {
        k: args2dict(v) if is_args(v) else v
        for k, v in vars(namespace).items()
    }