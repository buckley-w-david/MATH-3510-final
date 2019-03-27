import math
from jitcdde import jitcdde, y, t
import numpy

tau = 5.0
p1 = 0.35
y0 = 114
real_data = [114, 116, 114, 108, 112, 107, 108, 128, 169, 201, 212, 214, 245, 262, 297, 314, 340]

def E(t):
     if abs(t) < 1:
         return math.exp(-1 / (1 - t**2))
     else:
         return 0

def initial(t):
    value = [500*E(t+1.5)]
    # print(t, value)
    return value

f = [ p1 * y(0, t-tau) ]
DDE = jitcdde(f, max_delay=tau, verbose=True)
DDE.past_from_function(initial)

# Breaking encapsulation in order to set initial value
DDE.generate_lambdas()
DDE.DDE.y = numpy.array([y0])
DDE.DDE.past[-1] = (0.0, numpy.array([y0]), numpy.array([0.]))

data = []
range = DDE.t + numpy.arange(0, 13, 0.5)
for time in range:
    data.append( DDE.integrate(time) )

import matplotlib.pyplot as plt
plt.scatter(range, numpy.array(data))
plt.scatter(numpy.arange(0, 8.5, 0.5), real_data)
plt.show()
