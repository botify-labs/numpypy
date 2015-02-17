Welcome to PyPy's fork of Numpy. In order to install::

   git clone https://bitbucket.org/pypy/numpy.git; 
   cd numpy; pypy setup.py install

or more cleanly::

    pip install git+https://bitbucket.org/pypy/numpy.git

If you get a message about `missing Python.h` you must install the pypy-dev
package for your system

If you installed to a system directory, you may need to run::

    sudo pypy -c 'import numpy'

once to initialize the cffi cached shared objects as `root`

For now, NumPyPy only works with Python 2, and is not complete. You may get warnings or NotImplemented errors. Please let us know if you get crashes or wrong results.

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