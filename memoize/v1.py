import inspect
def memoize(obj):
    cache = obj.cache = {}
    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        kwargs.update(dict(zip(inspect.getargspec(obj).args, args)))
        key = tuple(kwargs.get(k, None) for k in inspect.getargspec(obj).args)
        if key not in cache:
            cache[key] = obj(**kwargs)
        return cache[key]
    return memoizer
