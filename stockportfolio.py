import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

innovations = np.random.weibull(2, 1000)*2.5
# innovations = np.random.standard_normal(10000)/10

trend = (1.07**(1/250)-1)

print('Daily trend: '+ str(trend))

innovations = innovations.copy() - np.mean(innovations) + trend + 0.035

stock = np.cumsum(innovations) + 100

returns = pd.DataFrame(stock).pct_change()

returns = np.array(returns)

returns = np.nan_to_num(returns)

print(['mean', np.mean(returns)])
print(['Std deviation: ', returns.std()])

fig, (axs1, axs2) = plt.subplots(2)
axs1.hist(returns, 'auto')
axs2.plot(stock)
plt.show()
# pr