
def NotImplementedFunc(func):
    def tmp(*args, **kwargs):
        raise NotImplementedError("%s not implemented yet" % func)
    return tmp

for name in '''
_insert add_docstring digitize bincount interp add_newdoc_ufunc
ravel_multi_index unravel_index packbits unpackbits
'''.split():
    if name not in globals():
        globals()[name] = NotImplementedFunc(name)
    else:
        print '_compiled_base now implements %s, please remove from lib/_compiled_base list of NotImplementedFuncs' % name
