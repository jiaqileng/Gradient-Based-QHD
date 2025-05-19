import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import seaborn as sns

from GradBasedQHD import GradBasedQHD

from numpy import sin, cos, sqrt, pi, exp


'''
Figure 4: Michalewicz function
'''
lb = 0
rb = pi 
N = 128 
success_gap = 1
s = lambda t: 10
beta = lambda t: -0.05 # 0.01 * np.sqrt(s(t))

def f(X,Y):
    return - sin(X) * sin(X**2/pi)**20 - sin(Y) * sin(2 * Y**2/pi)**20

def du_dx(X,Y):
    return - (1 / pi) * sin(X**2/pi)**19 * (40*X*sin(X)*cos(X**2/pi) + pi*sin(X**2/pi)*cos(X))

def du_dy(X,Y):
    return - (1 / pi) * sin(2*Y**2/pi)**19 * (80*Y*sin(Y)*cos(2*Y**2/pi) + pi*sin(2*Y**2/pi)*cos(Y))

grad = [du_dx, du_dy]

model = GradBasedQHD(f, grad, lb, rb, N, success_gap, s, beta)
fmin = model.get_fmin()

T = 10
snapshot_times_0, obj_val_0, success_prob_0  = model.qhd_simulator(T, 1000, 1)
snapshot_times_1, obj_val_1, success_prob_1 = model.high_res_qhd_simulator(T, 1000, 1)
# snapshot_times_2, obj_val_2, success_prob_2 = model.gd_samples(1000, 1000, 1e-2) 
snapshot_times_3, obj_val_3, success_prob_3 = model.nesterov_samples(1000, 1000, 1e-2) 
snapshot_times_4, obj_val_4, success_prob_4 = model.sgd_momentum_samples(1000, 1000, 1e-2)

iter_number = np.linspace(1, 1000, 1000)
y_data_0 = obj_val_0 - fmin
y_data_1 = obj_val_1 - fmin
y_data_3 = obj_val_3 - fmin
y_data_4 = obj_val_4 - fmin
Y_DATA_3 = [y_data_4, y_data_3, y_data_0, y_data_1]
Y_DATA_4 = [success_prob_4, success_prob_3, success_prob_0, success_prob_1]

# Make plot
LABELS = ['SGDM', 'NAG', 'QHD', 'Gradient-based QHD']
COLORS = ['limegreen', 'teal', 'skyblue', 'navy']

fig = plt.figure(figsize=(9, 9))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Serif",
    "text.latex.preamble": r"\usepackage{amsfonts}"
})
cm = sns.cubehelix_palette(start=.5, rot=-.5, reverse=True, as_cmap=True)

ax1 = fig.add_subplot(221, projection='3d')
ax1.view_init(elev=45, azim=30, roll=0)
surf = ax1.plot_surface(model.X, model.Y, model.V - fmin, 
                        rstride=1,cstride=1,cmap=cm,
                        linewidth=0, antialiased=False)
ax1.set_zlim(0, 2)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.zaxis.set_major_locator(LinearLocator(6))
ax2 = fig.add_subplot(222)
ax2.contourf(model.X, model.Y, model.V,cmap=cm)
ax2.plot([2.2], [1.57], marker='*', color='white', markersize=15)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax3 = fig.add_subplot(223)

for i in range(4):
    ax3.plot(iter_number, Y_DATA_3[i], label=LABELS[i], color=COLORS[i], linewidth=5)
ax3.legend(fontsize=12)
ax3.set_yscale('log')
ax3.set_xlabel('Iterations: ' + r'$k$', fontsize=16)
ax3.set_ylabel(r'$\mathbb{E}[f(X_k)] - f(x^*)$', fontsize=16)
ax3.set_ylim([2e-1, 2])
ax4 = fig.add_subplot(224)
for i in range(4):
    ax4.plot(iter_number, Y_DATA_4[i], label=LABELS[i], color=COLORS[i], linewidth=5)
ax4.legend(fontsize=12)
ax4.set_xlabel('Iterations: ' + r'$k$', fontsize=16)
ax4.set_ylabel(r'$\mathbf{P}_k$', fontsize=16)
ax4.set_ylim([0,1])
plt.savefig('figs/michalewicz.png', dpi=300, bbox_inches='tight')


'''
Figure 5: Cube-Wave function
'''
lb = -2
rb = 2
N = 128
success_gap = 1
s = lambda t: 10
beta = lambda t: -0.05 # 0.01 * np.sqrt(s(t))

def f(X,Y):
    return cos(pi*X)**2 + 0.25 * X**4 + cos(pi*Y)**2 + 0.25 * Y**4

def du_dx(X,Y):
    return -2 * pi * cos(pi*X) * sin(pi*X) + X**3

def du_dy(X,Y):
    return -2 * pi * cos(pi*Y) * sin(pi*Y) + Y**3

grad = [du_dx, du_dy]

model = GradBasedQHD(f, grad, lb, rb, N, success_gap, s, beta)
fmin = model.get_fmin()
T = 10

snapshot_times_0, obj_val_0, success_prob_0  = model.qhd_simulator(T, 500, 1)
snapshot_times_1, obj_val_1, success_prob_1 = model.high_res_qhd_simulator(T, 500, 1)
# snapshot_times_2, obj_val_2, success_prob_2 = model.gd_samples(1000, 1000, 1e-2) 
snapshot_times_3, obj_val_3, success_prob_3 = model.nesterov_samples(1000, 500, 2e-2) 
snapshot_times_4, obj_val_4, success_prob_4 = model.sgd_momentum_samples(1000, 500, 2e-2)

iter_number = np.linspace(1, 500, 500)
y_data_0 = obj_val_0 - fmin
y_data_1 = obj_val_1 - fmin
y_data_3 = obj_val_3 - fmin
y_data_4 = obj_val_4 - fmin
Y_DATA_3 = [y_data_4, y_data_3, y_data_0, y_data_1]
Y_DATA_4 = [success_prob_4, success_prob_3, success_prob_0, success_prob_1]

# Make plot
LABELS = ['SGDM', 'NAG', 'QHD', 'Gradient-based QHD']
COLORS = ['limegreen', 'teal', 'skyblue', 'navy']

fig = plt.figure(figsize=(9, 9))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Serif",
    "text.latex.preamble": r"\usepackage{amsfonts}"
})
cm = sns.cubehelix_palette(start=.5, rot=-.5, reverse=True, as_cmap=True)

ax1 = fig.add_subplot(221, projection='3d')
ax1.view_init(elev=45, azim=30, roll=0)
surf = ax1.plot_surface(model.X, model.Y, model.V, 
                        rstride=1,cstride=1,cmap=cm,
                        linewidth=0, antialiased=False)
ax1.set_zlim(0, 10)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.zaxis.set_major_locator(LinearLocator(6))
ax2 = fig.add_subplot(222)
ax2.contourf(model.X, model.Y, model.V,cmap=cm)
ax2.plot([-0.5], [-0.5], marker='*', color='white', markersize=15)
ax2.plot([-0.5], [0.5], marker='*', color='white', markersize=15)
ax2.plot([0.5], [-0.5], marker='*', color='white', markersize=15)
ax2.plot([0.5], [0.5], marker='*', color='white', markersize=15)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax3 = fig.add_subplot(223)
for i in range(4):
    ax3.plot(iter_number, Y_DATA_3[i], label=LABELS[i], color=COLORS[i], linewidth=5)
ax3.legend(fontsize=12)
ax3.set_yscale('log')
ax3.set_xlabel('Iterations: ' + r'$k$', fontsize=16)
ax3.set_ylabel(r'$\mathbb{E}[f(X_k)] - f(x^*)$', fontsize=16)
ax3.set_ylim([5e-4, 3])
ax4 = fig.add_subplot(224)
for i in range(4):
    ax4.plot(iter_number, Y_DATA_4[i], label=LABELS[i], color=COLORS[i], linewidth=5)
ax4.legend(fontsize=12)
ax4.set_xlabel('Iterations: ' + r'$k$', fontsize=16)
ax4.set_ylabel(r'$\mathbf{P}_k$', fontsize=16)
ax4.set_ylim([0,1.1])
plt.savefig('figs/cube-wave.png', dpi=300, bbox_inches='tight')


'''
Figure 6: Rastrigin function
'''
lb = -3
rb = 3
N = 128
success_gap = 1
s = lambda t: 10
beta = lambda t: -0.05

def f(X,Y):
    return X**2 - 10 * cos(2*pi*X) + Y**2 - 10 * cos(2*pi*Y) + 20
def du_dx(X,Y):
    return 2*X + 20*pi*sin(2*pi*X)

def du_dy(X,Y):
    return 2*Y + 20*pi*sin(2*pi*Y)

grad = [du_dx, du_dy]

model = GradBasedQHD(f, grad, lb, rb, N, success_gap, s, beta)
fmin = model.get_fmin()

T = 5
snapshot_times_0, obj_val_0, success_prob_0 = model.qhd_simulator(T, 1000, 1)
snapshot_times_1, obj_val_1, success_prob_1 = model.high_res_qhd_simulator(T, 1000, 1)
# snapshot_times_2, obj_val_2, success_prob_2 = model.gd_samples(1000, 1000, 5e-3) 
snapshot_times_3, obj_val_3, success_prob_3 = model.nesterov_samples(1000, 1000, 1e-3) 
snapshot_times_4, obj_val_4, success_prob_4 = model.sgd_momentum_samples(1000, 1000, 1e-3)

iter_number = np.linspace(1, 1000, 1000)
y_data_0 = obj_val_0 - fmin
y_data_1 = obj_val_1 - fmin
y_data_3 = obj_val_3 - fmin
y_data_4 = obj_val_4 - fmin
Y_DATA_3 = [y_data_4, y_data_3, y_data_0, y_data_1]
Y_DATA_4 = [success_prob_4, success_prob_3, success_prob_0, success_prob_1]

# Make plot
LABELS = ['SGDM', 'NAG', 'QHD', 'Gradient-based QHD']
COLORS = ['limegreen', 'teal', 'skyblue', 'navy']

fig = plt.figure(figsize=(9, 9))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Serif",
    "text.latex.preamble": r"\usepackage{amsfonts}"
})
cm = sns.cubehelix_palette(start=.5, rot=-.5, reverse=True, as_cmap=True)
ax1 = fig.add_subplot(221, projection='3d')
ax1.view_init(elev=45, azim=30, roll=0)
surf = ax1.plot_surface(model.X, model.Y, model.V, 
                        rstride=1,cstride=1,cmap=cm,
                        linewidth=0, antialiased=False)
ax1.set_zlim(0, 50)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.zaxis.set_major_locator(LinearLocator(6))
ax2 = fig.add_subplot(222)
ax2.contourf(model.X, model.Y, model.V,cmap=cm)
ax2.plot([0], [0], marker='*', color='white', markersize=15)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax3 = fig.add_subplot(223)
for i in range(4):
    ax3.plot(iter_number, Y_DATA_3[i], label=LABELS[i], color=COLORS[i], linewidth=5)

ax3.legend(fontsize=10)
ax3.set_yscale('log')
ax3.set_xlabel('Iterations: ' + r'$k$', fontsize=16)
ax3.set_ylabel(r'$\mathbb{E}[f(X_k)] - f(x^*)$', fontsize=16)
ax3.set_ylim([2e-1,30])
ax4 = fig.add_subplot(224)
for i in range(4):
    ax4.plot(iter_number, Y_DATA_4[i], label=LABELS[i], color=COLORS[i], linewidth=5)
ax4.legend(fontsize=10)
ax4.set_xlabel('Iterations: ' + r'$k$', fontsize=16)
ax4.set_ylabel(r'$\mathbf{P}_k$', fontsize=16)
plt.savefig('figs/rastrigin.png', dpi=300, bbox_inches='tight')