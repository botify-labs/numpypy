for name in '''
cffti cfftf
'''.split():
    if name not in globals():
        globals()[name] = None
