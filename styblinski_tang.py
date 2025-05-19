import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import seaborn as sns

from GradBasedQHD import GradBasedQHD

from numpy import sin, cos, sqrt, pi, exp


# Figure 1 & 2: Styblinski-Tang function

lb = -5
rb = 5
s = lambda t: 100
beta = lambda t: - 0.03 #lambda t: 5e-3 * np.sqrt(s(t))
success_gap = 1
N = 128

c = 0.2
def f(X,Y):
    return c * (X**4 - 16 * X**2 + 5 * X) + \
            c * (Y**4 - 16 * Y**2 + 5 * Y)

def df_dx(X,Y):
    return c * (4 * X**3 - 32*X+ 5)

def df_dy(X,Y):
    return c * (4 * Y**3 - 32*Y + 5)

grad = [df_dx, df_dy]


model = GradBasedQHD(f, grad, lb, rb, N, success_gap, s, beta)
fmin = model.get_fmin()
T = 5

snapshot_times_0, obj_val_0, success_prob_0, wave_fun_qhd = model.qhd_simulator(T, 500, 1, return_wave_fun=True)
snapshot_times_1, obj_val_1, success_prob_1, wave_fun_high_res_qhd = model.high_res_qhd_simulator(T, 500, 1, 200, return_wave_fun=True)
snapshot_times_3, obj_val_3, success_prob_3 = model.nesterov_samples(1000, 500, 1e-2)
snapshot_times_5, obj_val_5, success_prob_5 = model.sgd_momentum_samples(1000, 500, 1e-2)

y_data_0 = obj_val_0 - fmin
y_data_1 = obj_val_1 - fmin
y_data_3 = obj_val_3 - fmin
y_data_5 = obj_val_5 - fmin


# Figure 1: comparison between QHD and high-res-QHD
fig = plt.figure(figsize=(19,6))
gs = fig.add_gridspec(2, 6)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Serif",
    "text.latex.preamble": r"\usepackage{amssymb}"
})

cm_1 = sns.cubehelix_palette(start=.5, rot=-.5, reverse=True, as_cmap=True)
cm_2 = sns.color_palette("rocket", as_cmap=True)

ax0 = fig.add_subplot(gs[:2, :2], projection='3d')
ax0.view_init(elev=55)
surf = ax0.plot_surface(model.X, model.Y, model.V - fmin,
                       rstride=1,cstride=1,cmap=cm_1,
                        linewidth=0, antialiased=False)
ax0.set_zlim(0, 100)
ax0.set_xlabel('x')
ax0.set_ylabel('y')

ax1 = fig.add_subplot(gs[0, 2:6])
ax1.set_title('QHD', fontsize=18)
ax1.set_axis_off()

ax2 = fig.add_subplot(gs[1,2:6])
ax2.set_title('Gradient-Based QHD', fontsize=18)
ax2.set_axis_off()

for i in range(4):
    ax = fig.add_subplot(gs[0, 2+i], projection='3d')
    prob_qhd = abs(wave_fun_qhd[50*(i+1)])**2
    surf = ax.plot_surface(model.X, model.Y, prob_qhd, 
                           rstride=1,cstride=1,cmap=cm_2,
                           linewidth=0, antialiased=False)
    ax.text2D(-0.02, 0.07, r"$k = $" + f" {50 * (i+1)}", fontsize=14)
    ax.text2D(-0.03, 0.05, r"$\mathbf{P}_k = $" + f" {success_prob_0[50*(i+1)]:.2f}", fontsize=14)
    ax.set_axis_off()

    ax = fig.add_subplot(gs[1, 2+i], projection='3d')
    prob_high_res_qhd = abs(wave_fun_high_res_qhd[50*(i+1)])**2
    surf = ax.plot_surface(model.X, model.Y, prob_high_res_qhd,
                           rstride=1,cstride=1,cmap=cm_1,
                           linewidth=0, antialiased=False)
    ax.text2D(-0.02, 0.07, r"$k = $" + f" {50 * (i+1)}", fontsize=14)
    ax.text2D(-0.03, 0.05, r"$\mathbf{P}_k = $" + f" {success_prob_1[50*(i+1)]:.2f}", fontsize=14)
    ax.set_axis_off()

plt.savefig('figs/fig1_st_evolution.png', dpi=300, bbox_inches='tight')


# Figure 2: function values & success probability
LABELS = ['SGDM', 'NAG', 'QHD', 'Gradient-based QHD']
COLORS = ['limegreen', 'teal', 'skyblue', 'navy']
Y_DATA_obj = [y_data_5, y_data_3, y_data_0, y_data_1]
Y_DATA_prob = [success_prob_5, success_prob_3, success_prob_0, success_prob_1]
iter_number = np.linspace(1, 500, 500)

# objective function value
fig = plt.figure(figsize=(6,6))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Serif",
    "text.latex.preamble": r"\usepackage{amsfonts}"
})
for i in range(4):
    plt.plot(iter_number, Y_DATA_obj[i], label=LABELS[i], color=COLORS[i], linewidth=5)
plt.legend(fontsize=18)
plt.yscale('log')
plt.xlabel('Iterations: ' + r'$k$', fontsize=16)
plt.ylabel(r'$\mathbb{E}[f(X_k)] - f(x^*)$', fontsize=16)
plt.savefig('figs/st_obj.png', dpi=300, bbox_inches='tight')

# success probability
fig = plt.figure(figsize=(6,6))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Serif",
    "text.latex.preamble": r"\usepackage{amssymb}"
})
for i in range(4):
    plt.plot(iter_number, Y_DATA_prob[i], label=LABELS[i], color=COLORS[i], linewidth=5)
plt.legend(fontsize=18)
plt.xlabel('Iterations: ' + r'$k$', fontsize=16)
plt.ylabel(r'$\mathbf{P}_k$', fontsize=16)
plt.savefig('figs/st_prob.png', dpi=300, bbox_inches='tight')