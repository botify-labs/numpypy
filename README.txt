Welcome to PyPy's fork of Numpy. In order to install, first
install pypy, hints are here http://pypy.org/download.html. Note this is 
a binary install, no lengthy translation or compilation necessary. Once you
have pypy working and feel comfortable using it, you can install our version
of the numpy module into a virtual environment in a separate directory::

   virtualenv -p /path/to/pypy/bin/pypy /directory/to/try/pypy-numpy
   git clone https://bitbucket.org/pypy/numpy.git;
   cd numpy; /director/to/try/pypy-numpy/bin/pypy setup.py install

or without a git checkout::

   virtualenv -p /path/to/pypy/bin/pypy /directory/to/try/pypy-numpy
    /directory/to/try/pypy-numpy/bin/pip install git+https://bitbucket.org/pypy/numpy.git

If you get a message about `missing Python.h` you must install the pypy-dev
package for your system

If you installed to a system directory, you may need to run::

    sudo pypy -c 'import numpy'

once to initialize the cffi cached shared objects as `root`

For now, NumPyPy only works with Python 2, and is not complete. You may get
warnings or NotImplemented errors. Please let us know if you get crashes or
wrong results.

If you do not have lapack/blas runtimes, it may take over 10 minutes to install,
since it needs to build a lapack compatability library.

----------------------------------------

The original README.txt follows:

NumPy is the fundamental package needed for scientific computing with Python.
This package contains:

    * a powerful N-dimensional array object
    * sophisticated (broadcasting) functions
    * tools for integrating C/C++ and Fortran code
    * useful linear algebra, Fourier transform, and random number capabilities.

It derives from the old Numeric code base and can be used as a replacement for Numeric. It also adds the features introduced by numarray and can be used to replace numarray.

More information can be found at the website:

http://www.numpy.org

After installation, tests can be run with:

python -c 'import numpy; numpy.test()'

The most current development version is always available from our
git repository:

http://github.com/numpy/numpy