import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
t = np.linspace(0, 100, 1000)
stationary = np.random.normal(0, 1, 1000)
non_stationary = np.cumsum(np.random.normal(0, 1, 1000))

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, stationary)
plt.title('Stationary Process: White Noise')
plt.xlabel('Time')
plt.ylabel('Value')

plt.subplot(2, 1, 2)
plt.plot(t, non_stationary)
plt.title('Non-Stationary Process: Random Walk')
plt.xlabel('Time')
plt.ylabel('Value')

plt.tight_layout()
plt.savefig("plots/distributions.png")
