import numpy as np 
from numpy.linalg import eigh
from scipy.integrate import odeint, solve_ivp 
from scipy.sparse import csc_matrix, kron, eye, diags
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt
import time


class HighResQHD2D:
    def __init__(self, f, grad, init_state, lb, rb, s, beta, N):
        self.f=f # a lambda function with 2 variables
        self.grad=grad # a 2-array of lambda functions
        self.init_state=init_state
        self.lb=lb 
        self.rb=rb
        self.s=s
        self.beta=beta 
        self.N=N 

        self.dx = (self.rb - self.lb) / self.N 
        self.x_data = np.linspace(self.lb, self.rb-self.dx, self.N)

        X, Y = np.meshgrid(self.x_data, self.x_data, sparse=False)
        self.X = X 
        self.Y = Y
        self.V = f(X,Y)

    @staticmethod
    def fdm_discretization(f, grad, lb, rb, N):
        L = rb - lb
        dx = L / N
        x_data = np.linspace(lb, rb-dx, N)
        X, Y = np.meshgrid(x_data, x_data, indexing='ij')
        
        PP = np.diag(np.ones(N-1), 1) - np.diag(np.ones(N-1), -1)
        PP[0,-1] = -1
        PP[-1,0] = 1

        KK = np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1) - 2 * np.eye(N)
        KK[0,-1] = 1
        KK[-1,0] = 1

        P = - 1j / (2 * dx) * PP
        K = -0.5 / (dx**2) * KK

        # objective
        # V = np.diag(f(x_data))
        # G = np.diag(grad(x_data))
        obj = f(X, Y).flatten()
        g1 = grad[0](X, Y).flatten()
        g2 = grad[1](X, Y).flatten()

        return P, K, obj, g1, g2
    

    @staticmethod
    def dft_discretization(f, grad, lb, rb, N):
        L = rb - lb
        dx = L / N
        x_data = np.linspace(lb, rb-dx, N)
        X, Y = np.meshgrid(x_data, x_data, indexing='ij')

        DFT = np.fft.fft(np.eye(N)) / np.sqrt(N)
        wave_number = np.concatenate((np.linspace(0, N/2-1, int(N/2)), np.linspace(-N/2, -1, int(N/2))))
        d1 = 2 * np.pi * 1j * wave_number / L 
        d2 = -4 * np.pi**2 * wave_number**2 / L**2
        D1 = DFT.conj().T @ np.diag(d1) @ DFT
        D2 = DFT.conj().T @ np.diag(d2) @ DFT
        P = - 1j * csc_matrix(D1)
        K = -0.5 * csc_matrix(D2)

        # objective
        # V = csc_matrix(np.diag(f(x_data)))
        # G = csc_matrix(np.diag(grad(x_data)))
        obj = f(X, Y).flatten()
        g1 = grad[0](X, Y).flatten()
        g2 = grad[1](X, Y).flatten()

        return P, K, obj, g1, g2
    

    @staticmethod
    def construct_init_state(x_data, init_state):
        psi0 = np.array(init_state(x_data), dtype=complex)
        psi0 = np.kron(psi0, psi0)
        psi0 = psi0 / np.linalg.norm(psi0)
        
        return psi0
    

    def discretize(self):
        self.P, self.K, self.obj, self.g1, self.g2 = self.fdm_discretization(self.f, self.grad, self.lb, self.rb, self.N)
    
    
    # def Ham(self, t, y):
    #     P, K, V, G, ob, g = self.fdm_discretization(self.f, self.grad, self.lb, self.rb, self.N)
        
    #     psi = y[0:self.N] + 1j * y[self.N:2*self.N]
    #     a1 = (1 / t**3)
    #     a2 = 0.5 * self.beta 
    #     a3 = self.beta * (self.beta + np.sqrt(self.s)) * t**3
    #     a4 = t**3 + 1.5 * (2*self.beta + np.sqrt(self.s)) * t**2

    #     part_1 = K @ psi
    #     part_2 = 0.5 * (G @ P + P @ G) @ psi
    #     part_3 = (g**2) * psi
    #     part_4 = ob * psi
    #     dpdt = -1j * (a1 * part_1 + a2 * part_2 + a3 * part_3 + a4 * part_4)
    #     dydt = np.concatenate((np.real(dpdt), np.imag(dpdt)))
        
    #     return dydt 


    def run_qhd_2d(self, t0, tf, n_steps):
        P, K, obj, g1, g2 = self.fdm_discretization(self.f, self.grad, self.lb, self.rb, self.N)
        # P, K, obj, g1, g2 = self.dft_discretization(self.f, self.grad, self.lb, self.rb, self.N)

        H_kinetic = csc_matrix(kron(K, eye(self.N)) + kron(eye(self.N), K))

        def Ham(t):
            a1 = (1 / t**3)
            a4 = t**3

            H1 = a1 * H_kinetic
            h = a4 * obj
            return H1, h

        psi = self.construct_init_state(self.x_data, self.init_state)
        snapshot_times = np.linspace(t0, tf, n_steps)
        probability = np.zeros((n_steps, self.N**2))
        probability[0] = np.abs(psi)**2
        
        for i in range(n_steps-1):
            dt = snapshot_times[i+1] - snapshot_times[i]
            H1, h = Ham(snapshot_times[i])
            psi = expm_multiply(-1j*dt*H1, psi)
            psi = np.exp(-1j*dt*h) * psi
            probability[i+1] = np.abs(psi)**2
        
        return snapshot_times, probability


    def simulator_2d(self, t0, tf, n_steps):
        P, K, obj, g1, g2 = self.fdm_discretization(self.f, self.grad, self.lb, self.rb, self.N)
        G1 = np.diag(g1)
        G2 = np.diag(g2)
        P1 = kron(P,eye(self.N))
        P2 = kron(eye(self.N),P)

        H_kinetic = csc_matrix(kron(K, eye(self.N)) + kron(eye(self.N), K))
        H_cross = csc_matrix(G1 @ P1 + P1 @ G1 + G2 @ P2 + P2 @ G2)
        H_obj = csc_matrix(np.diag(obj))

        def Ham(t):
            a1 = (1 / t**3)
            a2 = 0.5 * self.beta 
            a3 = self.beta * (self.beta + np.sqrt(self.s)) * t**3
            a4 = t**3 + 1.5 * (2*self.beta + np.sqrt(self.s)) * t**2

            # H1 = a1 * H_kinetic
            # H2 = a2 * H_cross
            H = a1 * H_kinetic + a2 * H_cross + a4 * H_obj
            h = a3 * (g1**2 + g2**2) + a4 * obj
            return H, h

        # def Ham(t):
        #     a1 = (1 / t**3)
        #     a2 = 0.5 * self.beta 
        #     a3 = self.beta * (self.beta + np.sqrt(self.s)) * t**3
        #     a4 = t**3 + 1.5 * (2*self.beta + np.sqrt(self.s)) * t**2

        #     H1 = a1 * csc_matrix(kron(K, eye(self.N)) + kron(eye(self.N), K))
        #     H2 = a2 * csc_matrix(G1 @ P1 + P1 @ G1 + G2 @ P2 + P2 @ G2)
        #     h3 = a3 * (g1**2 + g2**2) + a4 * obj
        #     return H1, H2, h3

        # t0 = 1.5 * np.sqrt(self.s)
        # t_span = [t0, tf]
        psi = self.construct_init_state(self.x_data, self.init_state)
        snapshot_times = np.linspace(t0, tf, n_steps)
        probability = np.zeros((n_steps, self.N**2))
        probability[0] = np.abs(psi)**2
        
        for i in range(n_steps-1):
            dt = snapshot_times[i+1] - snapshot_times[i]
            H, h = Ham(snapshot_times[i])
            psi = expm_multiply(-1j*dt*H, psi)
            # psi = expm_multiply(-1j*dt*H2, psi)
            # psi = np.exp(-1j*dt*h) * psi
            probability[i+1] = np.abs(psi)**2
        
        return snapshot_times, probability
    

    def simulator_2d_new(self, t0, tf, n_steps):
        P, K, obj, g1, g2 = self.fdm_discretization(self.f, self.grad, self.lb, self.rb, self.N)
        G1 = np.diag(g1)
        G2 = np.diag(g2)
        P1 = kron(P,eye(self.N))
        P2 = kron(eye(self.N),P)

        H_kinetic = csc_matrix(kron(K, eye(self.N)) + kron(eye(self.N), K))
        H_cross = csc_matrix(G1 @ P1 + P1 @ G1 + G2 @ P2 + P2 @ G2)

        def Ham(t):
            a1 = (1 / t**3)
            a2 = 0.5 * self.beta 
            a3 = 0.5 * self.beta * np.sqrt(self.s) * t**3
            a4 = t**3 + (3*self.beta + 1.5*np.sqrt(self.s)) * t**2

            # H1 = a1 * H_kinetic
            # H2 = a2 * H_cross
            H = a1 * H_kinetic + a2 * H_cross
            h = a3 * (g1**2 + g2**2) + a4 * obj
            return H, h

        # def Ham(t):
        #     a1 = (1 / t**3)
        #     a2 = 0.5 * self.beta 
        #     a3 = self.beta * (self.beta + np.sqrt(self.s)) * t**3
        #     a4 = t**3 + 1.5 * (2*self.beta + np.sqrt(self.s)) * t**2

        #     H1 = a1 * csc_matrix(kron(K, eye(self.N)) + kron(eye(self.N), K))
        #     H2 = a2 * csc_matrix(G1 @ P1 + P1 @ G1 + G2 @ P2 + P2 @ G2)
        #     h3 = a3 * (g1**2 + g2**2) + a4 * obj
        #     return H1, H2, h3

        # t0 = 1.5 * np.sqrt(self.s)
        # t_span = [t0, tf]
        psi = self.construct_init_state(self.x_data, self.init_state)
        snapshot_times = np.linspace(t0, tf, n_steps)
        probability = np.zeros((n_steps, self.N**2))
        probability[0] = np.abs(psi)**2
        
        for i in range(n_steps-1):
            dt = snapshot_times[i+1] - snapshot_times[i]
            H, h = Ham(snapshot_times[i])
            psi = expm_multiply(-1j*dt*H, psi)
            # psi = expm_multiply(-1j*dt*H2, psi)
            psi = np.exp(-1j*dt*h) * psi
            probability[i+1] = np.abs(psi)**2
        
        return snapshot_times, probability
    

    def simulator_2d_timing(self, t0, tf, n_steps):
        # P, K, obj, g1, g2 = self.fdm_discretization(self.f, self.grad, self.lb, self.rb, self.N)
        P, K, obj, g1, g2 = self.dft_discretization(self.f, self.grad, self.lb, self.rb, self.N)
        G1 = np.diag(g1)
        G2 = np.diag(g2)
        P1 = kron(P,eye(self.N))
        P2 = kron(eye(self.N),P)

        H_kinetic = csc_matrix(kron(K, eye(self.N)) + kron(eye(self.N), K))
        H_cross = csc_matrix(G1 @ P1 + P1 @ G1 + G2 @ P2 + P2 @ G2)

        def Ham(t):
            a1 = (1 / t**3)
            a2 = 0.5 * self.beta 
            a3 = self.beta * (self.beta + np.sqrt(self.s)) * t**3
            a4 = t**3 + 1.5 * (2*self.beta + np.sqrt(self.s)) * t**2

            H1 = a1 * H_kinetic
            H2 = a2 * H_cross
            h3 = a3 * (g1**2 + g2**2) + a4 * obj
            return H1, H2, h3

        # t0 = 1.5 * np.sqrt(self.s)
        # t_span = [t0, tf]
        psi = self.construct_init_state(self.x_data, self.init_state)
        snapshot_times = np.linspace(t0, tf, n_steps)
        probability = np.zeros((n_steps, self.N**2))
        probability[0] = np.abs(psi)**2
        
        dt = snapshot_times[1] - snapshot_times[0]

        start_0 = time.time()
        H1, H2, h3 = Ham(snapshot_times[0])
        end_0 = time.time()

        start_1 = time.time()
        psi = expm_multiply(-1j*dt*H1, psi)
        end_1 = time.time()

        start_2 = time.time()
        psi = expm_multiply(-1j*dt*H2, psi)
        end_2 = time.time()

        start_3 = time.time()
        psi = np.exp(-1j*dt*h3) * psi
        end_3 = time.time()

        start_4 = time.time()
        psi = expm_multiply(-1j*dt*(H1+H2), psi)
        end_4 = time.time()

        print(f"matrix formation: {end_0 - start_0}")
        print(f"H1: {end_1 - start_1}")
        print(f"H2: {end_2 - start_2}")
        print(f"h: {end_3 - start_3}")
        print(f"H1+H2: {end_4 - start_4}")
        
        return


    # def simulator_legacy(self, tf, n_steps):
    #     P, K, V, G, ob, g = self.fdm_discretization(self.f, self.grad, self.lb, self.rb, self.N)

    #     def Ham(t, y):
    #         psi = y[0:self.N] + 1j * y[self.N:2*self.N]
    #         a1 = (1 / t**3)
    #         a2 = 0.5 * self.beta 
    #         a3 = self.beta * (self.beta + np.sqrt(self.s)) * t**3
    #         a4 = t**3 + 1.5 * (2*self.beta + np.sqrt(self.s)) * t**2

    #         part_1 = K @ psi
    #         part_2 = 0.5 * (G @ P + P @ G) @ psi
    #         part_3 = (g**2) * psi
    #         part_4 = ob * psi
    #         dpdt = -1j * (a1 * part_1 + a2 * part_2 + a3 * part_3 + a4 * part_4)
    #         dydt = np.concatenate((np.real(dpdt), np.imag(dpdt)))
            
    #         return dydt 

    #     t0 = 1.5 * np.sqrt(self.s)
    #     t_span = [t0, tf]
    #     psi0 = self.construct_init_state(self.x_data)
    #     y0 = np.concatenate((np.real(psi0), np.imag(psi0)))
    #     snapshot_times = np.linspace(t0, tf, n_steps)
    #     sol = solve_ivp(Ham, t_span, y0, t_eval=snapshot_times)
    #     # sol = odeint(Ham, y0, snapshot_times)

    #     probability = np.zeros((n_steps, self.N))
    #     for i in range(n_steps):
    #         prob = np.abs(sol.y[0:self.N,i])**2 + np.abs(sol.y[self.N:2*self.N,i])**2
    #         # prob = sol[i,0:self.N]**2 + sol[i,self.N:2*self.N]**2
    #         probability[i] = prob
        
    #     return snapshot_times, probability
    

    def compute_spectral_gap_qhd(self, t0, tf, n_steps):
        snapshot_times = np.linspace(t0, tf, n_steps)
        P, K, V, G, ob, g = self.fdm_discretization(self.f, self.grad, self.lb, self.rb, self.N)

        def Ham(t):
            a1 = (1 / t**3)
            a4 = t**3

            H1 = K
            H4 = V 
            H = a1 * H1 + a4 * H4
            return H
        
        spectral_gap = np.zeros_like(snapshot_times)
        for i in range(n_steps):
            t_point = snapshot_times[i]
            eigenvalues, _ = eigh(Ham(t_point))
            spectral_gap[i] = eigenvalues[1] - eigenvalues[0]

        return snapshot_times, spectral_gap


    def compute_spectral_gap_highres_qhd(self, t0, tf, n_steps):
        snapshot_times = np.linspace(t0, tf, n_steps)
        P, K, V, G, ob, g = self.fdm_discretization(self.f, self.grad, self.lb, self.rb, self.N)

        def Ham(t):
            a1 = (1 / t**3)
            a2 = 0.5 * self.beta 
            a3 = self.beta * (self.beta + np.sqrt(self.s)) * t**3
            a4 = t**3 + 1.5 * (2*self.beta + np.sqrt(self.s)) * t**2

            H1 = K
            H2 = 0.5 * (G @ P + P @ G)
            H3 = np.diag(g**2)
            H4 = V 
            H = a1 * H1 + a2 * H2 + a3 * H3 + a4 * H4
            return H
        
        spectral_gap = np.zeros_like(snapshot_times)
        for i in range(n_steps):
            t_point = snapshot_times[i]
            eigenvalues, _ = eigh(Ham(t_point))
            spectral_gap[i] = eigenvalues[1] - eigenvalues[0]

        return snapshot_times, spectral_gap




if __name__ == '__main__':
    # f = lambda x: x**4 - (x - 1/8)**2
    # grad = lambda x: 4 * x**3 - 2 * (x - 1/8)
    f = lambda x: (4 * x - 2)**4 - (4 * x - 2 - 1/8)**2
    grad = lambda x: 16 * (4 * x - 2)**3 - 8 * (4 * x - 2 - 1/8)
    lb = 0
    rb = 1
    s = 0.05
    beta = 0.1 * np.sqrt(s)
    N = 64

    model = HighResQHD(f, grad, lb, rb, s, beta, N)
    snapshot_times, probability = model.simulator(10, 100)
    

    # for i in range(10):
        # plt.plot(model.x_data, probability[10*i])
        # print(np.sum(probability[10*i]))
    plt.plot(model.x_data, probability[-1])
    plt.show()

