class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def load_args(f):
    import yaml
    with open(f) as fp:
        args = AttributeDict(yaml.load(fp))
    return args
