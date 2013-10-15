for name in '''
cffti cfftf
'''.split():
    assert name not in globals()
    globals()[name] = None
