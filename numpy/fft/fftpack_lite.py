
def NotImplementedFunc(func):
    def tmp(*args, **kwargs):
        raise NotImplementedError("%s not implemented yet" % func)
    return tmp

for name in '''
cffti cfftf cfftb rffti rfftb rfftf
'''.split():
    if name not in globals():
        globals()[name] = None
