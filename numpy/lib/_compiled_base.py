for name in '''
_insert add_docstring digitize bincount interp add_newdoc_ufunc
ravel_multi_index unravel_index packbits unpackbits
'''.split():
    if name not in globals():
        globals()[name] = None
