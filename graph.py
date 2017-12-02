import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print "Please specify a circuit type"
circuit = sys.argv[1]
npzfile = np.load('results.npz')
names = npzfile['arr_0']
values = npzfile['arr_1']
y_pos = np.arange(len(names))
plt.barh(y_pos, values, align='center')
plt.yticks(y_pos, names)
plt.title("R^2 Values for " + circuit + " Circuit")
plt.show()
