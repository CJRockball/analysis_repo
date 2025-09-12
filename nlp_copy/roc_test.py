#%%
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import beta, norm

x = np.linspace(0,1, 10000)
y = beta.pdf(x, a=0.5, b=0.5)

plt.figure()
plt.plot(x,y)
plt.show()



# %%


x = np.linspace(-3,7, 10000)
y = norm.pdf(x)
x1 = np.linspace(-3,7, 10000)
y1 = norm.pdf(x, 4,1)


plt.figure()
plt.plot(x,y)
plt.plot(x1,y1,color='orange')
plt.show()

# %%
