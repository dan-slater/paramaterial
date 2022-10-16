"""Module to make simple plot"""

# plot a decaying vibration
import matplotlib.pyplot as plt
import numpy as np

# create data
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-x / 10)

# plot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set(xlabel='time (s)', ylabel='vibration (m/s^2)',
         title='Decaying Vibration')
ax.grid()

plt.show()
