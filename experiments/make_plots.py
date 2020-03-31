
import pickle
import gradual_st.utils as utils
import matplotlib.pyplot as plt
from matplotlib import rc

font = {'family' : 'normal',
        'size'   : 16}
rc('font', **font)
binary_genders = pickle.load(open('portraits_gender_stats', 'rb'))
r = 1000
N = 18000
y = utils.rolling_average(binary_genders, r)[:N-r] * 100
x = np.array(list(range(N-r)))
yerr = 0.5 / np.sqrt(r) * 1.645 * 100
# plt.errorbar(
#     x, y, yerr=yerr,
#     barsabove=True, color='blue', capsize=4, linestyle='--')
plt.ylim([0, 100])
plt.plot(y)
plt.fill_between(x, y-yerr, y+yerr, facecolor='blue', alpha=0.2)
plt.xlabel("Time")
plt.ylabel("% Female")
plt.show()
