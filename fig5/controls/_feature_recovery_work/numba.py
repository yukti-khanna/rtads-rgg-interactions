def njit(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    def deco(fn):
        return fn
    return deco
