from numpy import *
from random import *
from pylab import *

if __name__ == '__main__':
    mat = matrix([[0.4, 0.5, 0.6],
    [0.3, 0.2, 0.1],
    [0.2, 0.5, 0.8]])

    pcolormesh(array(mat))
    colorbar
    show()
