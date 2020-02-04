
import pickle
import utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

font = {'family' : 'normal',
        'size'   : 16}
rc('font', **font)
binary_genders = pickle.load(open('portraits_gender_stats', 'rb'))
r = 1000
N = 18000
y = utils.rolling_average(binary_genders, r)[:N-r]
x = np.array(list(range(N-r)))
yerr = 0.5 / np.sqrt(r) * 1.645
# plt.errorbar(
#     x, y, yerr=yerr,
#     barsabove=True, color='blue', capsize=4, linestyle='--')
plt.ylim([0, 1])
plt.plot(y)
plt.fill_between(x, y-yerr, y+yerr, facecolor='blue', alpha=0.2)
plt.xlabel("Year")
plt.ylabel("% Female")
plt.show()
