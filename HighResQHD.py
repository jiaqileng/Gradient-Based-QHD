import numpy as np 
from numpy import pi, exp, sin, cos, sqrt
from numpy.fft import fft2, ifft2
from numpy.linalg import eigh, norm
from scipy.integrate import odeint, solve_ivp 
from scipy.sparse import csc_matrix, kron, eye, diags
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt


class HighResQHD:
    def __init__(self, f, grad, lb, rb, N, success_gap, s, beta, gamma=5):
        self.f=f
        self.grad=grad
        self.lb=lb 
        self.rb=rb
        self.N=N 
        self.success_gap=success_gap
        self.s=s
        self.beta=beta 
        self.gamma=gamma
        

        L = rb - lb
        self.L = L
        dx = L /N
        x_data= np.linspace(lb, rb-dx, N)
        X, Y = np.meshgrid(x_data, x_data, sparse=False)
        self.X = X 
        self.Y = Y

        # c = 2e-3
        # self.V = c * ((10*X-5)**4 - 16 * (10*X-5)**2 + 5 * (10*X-5)) + \
        #          c * ((10*Y-5)**4 - 16 * (10*Y-5)**2 + 5 * (10*Y-5))
        # self.Gx = 10*c * (4 * (10*X-5)**3 - 32*(10*X-5) + 5)
        # self.Gy = 10*c * (4 * (10*Y-5)**3 - 32*(10*Y-5) + 5)
        # self.G2 = self.Gx**2 + self.Gy**2

        self.V = f(X,Y)
        self.fmin = np.min(self.V[:])
        df_dx, df_dy = grad
        self.Gx = df_dx(X,Y)
        self.Gy = df_dy(X,Y)
        self.G2 = self.Gx**2 + self.Gy**2

        wave_number = np.concatenate((np.linspace(0, N/2-1, int(N/2)), np.linspace(-N/2, -1, int(N/2))))
        k_1, k_2 = np.meshgrid(wave_number, wave_number, sparse=True)
        
        self.dx_coeff = 2*pi*1j / L * k_1 + 0*k_2
        self.dy_coeff = 2*pi*1j / L * k_2 + 0*k_1
        self.d2_coeff = - 4 * pi**2 / L**2 * (k_1**2 + k_2**2)


    @staticmethod
    def construct_init_state(N):
        u0 = np.ones((N, N))
        u0 /= np.sqrt(np.sum(u0**2)) 
        return u0
    
    def success_indicator(self):
        indicator = self.V - self.fmin < self.success_gap
        return indicator


    def qhd_simulator(self, T, n_iter, capture_every, return_wave_fun=False, T0=0):
        snapshot_times = []
        obj_val = []
        success_prob = []
        wave_fun = []

        dt = (T - T0) / n_iter
        psi = self.construct_init_state(self.N)
        success_indicator = self.success_indicator()
        tdep1 = lambda t: 1 / t**3
        tdep2 = lambda t: t**3

        for i in range(n_iter):
            t_temp = T0 + dt * (1+i)
            psi = exp(-1j * dt * tdep2(t_temp) * self.V) * psi 
            psi = ifft2(exp(1j*dt*tdep1(t_temp) * self.d2_coeff) * fft2(psi))

            if i % capture_every == 0:
                prob = np.abs(psi)**2
                prob /= np.sum(prob)
                snapshot_times.append(t_temp)
                obj_val.append(np.sum(prob * self.V))
                success_prob.append(np.sum(prob * success_indicator))
                if return_wave_fun:
                    wave_fun.append(psi)
            
        # add the last frame if not before
        if i % capture_every != 0:
            prob = np.abs(psi)**2
            prob /= np.sum(prob)
            snapshot_times.append(t_temp)
            obj_val.append(np.sum(prob * self.V))
            success_prob.append(np.sum(prob * success_indicator))
            if return_wave_fun:
                wave_fun.append(psi)

        if return_wave_fun:
            return np.array(snapshot_times), np.array(obj_val), np.array(success_prob), wave_fun
        else:
            return np.array(snapshot_times), np.array(obj_val), np.array(success_prob)
    

    def high_res_qhd_helper(self, t0, t1, inner_loop_iter, psi):
        t = (t1 + t0) / 2
        dt = (t1 - t0) / inner_loop_iter

        s_temp = self.s(t)
        beta_temp = self.beta(t)
        a1 = (1 / t**3)
        a2 = 0.5 * beta_temp
        a3 = 0 # 0.5 * beta_temp * (beta_temp + sqrt(s_temp)) * t**3
        a4 = t**3 + self.gamma * t**2
        #t**3 + 1.5 * (2 * beta_temp + sqrt(s_temp)) * t**2

        for j in range(inner_loop_iter):
            psi = exp(-1j * dt / 2 * (a3 * self.G2 + a4 * self.V)) * psi 
            psi = ifft2(exp(1j * dt / 2 * a1 * self.d2_coeff) * fft2(psi))
            psi -= a2 * dt* (self.Gx * ifft2(self.dx_coeff * fft2(psi)) + ifft2(self.dx_coeff * fft2(self.Gx * psi)))
            psi -= a2 * dt* (self.Gy * ifft2(self.dy_coeff * fft2(psi)) + ifft2(self.dy_coeff * fft2(self.Gy * psi)))
            psi = exp(-1j * dt / 2 * (a3 * self.G2 + a4 * self.V)) * psi 
            psi = ifft2(exp(1j * dt / 2 * a1 * self.d2_coeff) * fft2(psi))
        
        return psi




    def high_res_qhd_simulator(self, T, n_iter, capture_every, inner_loop_iter=100, return_wave_fun=False, T0=0):
        snapshot_times = []
        obj_val = []
        success_prob = []
        wave_fun = []

        dt = (T - T0) / n_iter
        psi = self.construct_init_state(self.N)
        success_indicator = self.success_indicator()

        for i in range(n_iter):
            t0 = T0 + dt * i
            t1 = t0 + dt
            psi = self.high_res_qhd_helper(t0, t1, inner_loop_iter, psi)
            # s_temp = self.s(t)
            # beta_temp = self.beta(t)

            # a1 = (1 / t**3)
            # a2 = 0.5 * beta_temp
            # a3 = 0.5 * beta_temp * (beta_temp + sqrt(s_temp)) * t**3
            # a4 = t**3 + 1.5 * (2 * beta_temp + sqrt(s_temp)) * t**2
            
            # psi = exp(-1j * dt / 2 * (a3 * self.G2 + a4 * self.V)) * psi 
            # psi = ifft2(exp(1j * dt / 2 * a1 * self.d2_coeff) * fft2(psi))
            # psi -= a2 * dt* (self.Gx * ifft2(self.dx_coeff * fft2(psi)) + ifft2(self.dx_coeff * fft2(self.Gx * psi)))
            # psi -= a2 * dt* (self.Gy * ifft2(self.dy_coeff * fft2(psi)) + ifft2(self.dy_coeff * fft2(self.Gy * psi)))
            # psi = exp(-1j * dt / 2 * (a3 * self.G2 + a4 * self.V)) * psi 
            # psi = ifft2(exp(1j * dt / 2 * a1 * self.d2_coeff) * fft2(psi))

            if i % capture_every == 0:
                prob = np.abs(psi)**2
                prob /= np.sum(prob)
                snapshot_times.append(t1)
                obj_val.append(np.sum(prob * self.V))
                success_prob.append(np.sum(prob * success_indicator))
                if return_wave_fun:
                    wave_fun.append(psi)


        # add the last frame if not before
        if i % capture_every != 0:
            prob = np.abs(psi)**2
            prob /= np.sum(prob)
            snapshot_times.append(t1)
            obj_val.append(np.sum(prob * self.V))
            success_prob.append(np.sum(prob * success_indicator))
            if return_wave_fun:
                wave_fun.append(psi)

        if return_wave_fun:
            return np.array(snapshot_times), np.array(obj_val), np.array(success_prob), wave_fun
        else:
            return np.array(snapshot_times), np.array(obj_val), np.array(success_prob)
    


    def high_res_qhd_simulator_legacy(self, T, n_iter, capture_every):
        snapshot_times = []
        obj_val = []
        success_prob = []

        dt = T / n_iter
        psi = self.construct_init_state(self.N)
        success_indicator = self.success_indicator()

        for i in range(n_iter):
            t = dt * (1+i)
            s_temp = self.s(t)
            beta_temp = self.beta(t)

            a1 = (1 / t**3)
            a2 = 0.5 * beta_temp
            a3 = 0.5 * beta_temp * (beta_temp + sqrt(s_temp)) * t**3
            a4 = t**3 + 1.5 * (2 * beta_temp + sqrt(s_temp)) * t**2
            
            psi = exp(-1j * dt / 2 * (a3 * self.G2 + a4 * self.V)) * psi 
            psi = ifft2(exp(1j * dt / 2 * a1 * self.d2_coeff) * fft2(psi))
            psi -= a2 * dt* (self.Gx * ifft2(self.dx_coeff * fft2(psi)) + ifft2(self.dx_coeff * fft2(self.Gx * psi)))
            psi -= a2 * dt* (self.Gy * ifft2(self.dy_coeff * fft2(psi)) + ifft2(self.dy_coeff * fft2(self.Gy * psi)))
            psi = exp(-1j * dt / 2 * (a3 * self.G2 + a4 * self.V)) * psi 
            psi = ifft2(exp(1j * dt / 2 * a1 * self.d2_coeff) * fft2(psi))

            if i % capture_every == 0:
                prob = np.abs(psi)**2
                prob /= np.sum(prob)
                snapshot_times.append(t)
                obj_val.append(np.sum(prob * self.V))
                success_prob.append(np.sum(prob * success_indicator))

        # add the last frame if not before
        if i % capture_every != 0:
            prob = np.abs(psi)**2
            prob /= np.sum(prob)
            snapshot_times.append(t)
            obj_val.append(np.sum(prob * self.V))
            success_prob.append(np.sum(prob * success_indicator))

        return np.array(snapshot_times), np.array(obj_val), np.array(success_prob)
    

    def get_fmin(self):
        return np.min(self.V)
    

    def gd_samples(self, n_samples, n_steps, lr):
        snapshot_times = np.linspace(0, (n_steps-1) * lr, n_steps)

        df_dx, df_dy = self.grad
        obj_val_mat = np.zeros((n_samples, n_steps))
        success_prob_mat = np.zeros((n_samples, n_steps))
        
        for i in range(n_samples):
            x = self.lb + np.random.rand(2) * self.L 

            for j in range(n_steps):
                current_f = self.f(x[0], x[1])
                obj_val_mat[i,j] = current_f 
                if current_f < self.success_gap:
                    success_prob_mat[i,j] = 1
                g = np.array([df_dx(x[0],x[1]), df_dy(x[0],x[1])])
                x -= lr * g 
                

        mean_obj_val = np.mean(obj_val_mat, axis=0)
        success_prob = np.mean(success_prob_mat, axis=0)
        return snapshot_times, mean_obj_val, success_prob
    

    def nesterov_samples(self, n_samples, n_steps, lr, return_grad_norm=False):
        snapshot_times = np.linspace(0, (n_steps-1) * lr, n_steps)

        df_dx, df_dy = self.grad
        obj_val_mat = np.zeros((n_samples, n_steps))
        success_prob_mat = np.zeros((n_samples, n_steps))

        if return_grad_norm:
            grad_norm_mat = np.zeros((n_samples, n_steps))
        
        for i in range(n_samples):
            x = self.lb + np.random.rand(2) * self.L 
            y = x

            for j in range(n_steps):
                current_f = self.f(x[0], x[1])
                obj_val_mat[i,j] = current_f 
                if current_f - self.fmin < self.success_gap:
                    success_prob_mat[i,j] = 1

                g1 = np.array([df_dx(y[0],y[1]), df_dy(y[0],y[1])])
                if return_grad_norm:
                    grad_norm_mat[i,j] = norm(g1)**2

                # NAGD
                x0 = x
                x = y - lr * g1 
                y = x + j / (j+3) * (x - x0)
                
        mean_obj_val = np.mean(obj_val_mat, axis=0)
        success_prob = np.mean(success_prob_mat, axis=0)
        
        if return_grad_norm:
            grad_norm = np.mean(grad_norm_mat, axis=0)
            return snapshot_times, mean_obj_val, success_prob, grad_norm
        else:
            return snapshot_times, mean_obj_val, success_prob
    

    def sgd_samples(self, n_samples, n_steps, lr, return_grad_norm=False):
        snapshot_times = np.linspace(0, (n_steps-1) * lr, n_steps)

        df_dx, df_dy = self.grad
        obj_val_mat = np.zeros((n_samples, n_steps))
        success_prob_mat = np.zeros((n_samples, n_steps))

        if return_grad_norm:
            grad_norm_mat = np.zeros((n_samples, n_steps))
        
        for i in range(n_samples):
            x = self.lb + np.random.rand(2) * self.L

            for j in range(n_steps):
                current_f = self.f(x[0], x[1])
                obj_val_mat[i,j] = current_f 
                if current_f - self.fmin < self.success_gap:
                    success_prob_mat[i,j] = 1

                g = np.array([df_dx(x[0],x[1]), df_dy(x[0],x[1])])
                if return_grad_norm:
                    grad_norm_mat[i,j] = norm(g)**2

                # SGD
                v = np.random.rand(2)
                x -= lr * g + lr / (j+1) * v
                
        mean_obj_val = np.mean(obj_val_mat, axis=0)
        success_prob = np.mean(success_prob_mat, axis=0)
        
        if return_grad_norm:
            grad_norm = np.mean(grad_norm_mat, axis=0)
            return snapshot_times, mean_obj_val, success_prob, grad_norm
        else:
            return snapshot_times, mean_obj_val, success_prob
        
    
    def sgd_momentum_samples(self, n_samples, n_steps, lr, return_grad_norm=False):
        snapshot_times = np.linspace(0, (n_steps-1) * lr, n_steps)

        df_dx, df_dy = self.grad
        obj_val_mat = np.zeros((n_samples, n_steps))
        success_prob_mat = np.zeros((n_samples, n_steps))

        if return_grad_norm:
            grad_norm_mat = np.zeros((n_samples, n_steps))
        
        
        increment_eta = 0.4 / n_steps
        for i in range(n_samples):
            x = self.lb + np.random.rand(2) * self.L
            v = np.zeros(2)

            for j in range(n_steps):
                current_f = self.f(x[0], x[1])
                obj_val_mat[i,j] = current_f 
                if current_f - self.fmin < self.success_gap:
                    success_prob_mat[i,j] = 1

                g = np.array([df_dx(x[0],x[1]), df_dy(x[0],x[1])])
                if return_grad_norm:
                    grad_norm_mat[i,j] = norm(g)**2

                # SGD with momentum
                n = np.random.rand(2)
                eta = 0.5 + (j+1) * increment_eta
                v = eta * v - (1 - eta) * lr * (g + 1 / (j+1) * n)
                x += v
                
        mean_obj_val = np.mean(obj_val_mat, axis=0)
        success_prob = np.mean(success_prob_mat, axis=0)
        
        if return_grad_norm:
            grad_norm = np.mean(grad_norm_mat, axis=0)
            return snapshot_times, mean_obj_val, success_prob, grad_norm
        else:
            return snapshot_times, mean_obj_val, success_prob
        

    # @staticmethod
    # def fdm_discretization(f, grad, lb, rb, N):
    #     L = rb - lb
    #     dx = L / N
    #     x_data = np.linspace(lb, rb-dx, N)
        
    #     PP = np.diag(np.ones(N-1), 1) - np.diag(np.ones(N-1), -1)
    #     KK = np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1) - 2 * np.eye(N)
    #     # D1 = 1 / (2 * dx) * PP
    #     # D2 = 1 / (dx**2) * KK
    #     # P = - 1j * csc_matrix(D1)
    #     # K = -0.5 * csc_matrix(D2)

    #     P = - 1j / (2 * dx) * PP
    #     K = -0.5 / (dx**2) * KK

    #     # objective
    #     V = np.diag(f(x_data))
    #     G = np.diag(grad(x_data))

    #     ob = f(x_data)
    #     g = grad(x_data)

    #     return P, K, V, G, ob, g
    

    # @staticmethod
    # def dft_discretization(f, grad, lb, rb, N):
    #     L = rb - lb
    #     dx = L / N
    #     x_data = np.linspace(lb, rb-dx, N)

    #     DFT = np.fft.fft(np.eye(N)) / np.sqrt(N)
    #     wave_number = np.concatenate((np.linspace(0, N/2-1, int(N/2)), np.linspace(-N/2, -1, int(N/2))))
    #     d1 = 2 * np.pi * 1j * wave_number / L 
    #     d2 = -4 * np.pi**2 * wave_number**2 / L**2
    #     D1 = DFT.conj().T @ np.diag(d1) @ DFT
    #     D2 = DFT.conj().T @ np.diag(d2) @ DFT
    #     P = - 1j * csc_matrix(D1)
    #     K = -0.5 * csc_matrix(D2)

    #     # objective
    #     V = csc_matrix(np.diag(f(x_data)))
    #     G = csc_matrix(np.diag(grad(x_data)))

    #     ob = f(x_data)
    #     g = grad(x_data)

    #     return P, K, V, G, ob, g
    

    # @staticmethod
    # def construct_init_state(x_data, init_state):
    #     psi0 = np.array(init_state(x_data), dtype=complex)
    #     psi0 = psi0 / np.linalg.norm(psi0)
        
    #     return psi0
    

    # def discretize(self):
    #     self.P, self.K, _, self.G, self.ob, self.g = self.fdm_discretization(self.f, self.grad, self.lb, self.rb, self.N)
    
    
    # # def Ham(self, t, y):
    # #     P, K, V, G, ob, g = self.fdm_discretization(self.f, self.grad, self.lb, self.rb, self.N)
        
    # #     psi = y[0:self.N] + 1j * y[self.N:2*self.N]
    # #     a1 = (1 / t**3)
    # #     a2 = 0.5 * self.beta 
    # #     a3 = self.beta * (self.beta + np.sqrt(self.s)) * t**3
    # #     a4 = t**3 + 1.5 * (2*self.beta + np.sqrt(self.s)) * t**2

    # #     part_1 = K @ psi
    # #     part_2 = 0.5 * (G @ P + P @ G) @ psi
    # #     part_3 = (g**2) * psi
    # #     part_4 = ob * psi
    # #     dpdt = -1j * (a1 * part_1 + a2 * part_2 + a3 * part_3 + a4 * part_4)
    # #     dydt = np.concatenate((np.real(dpdt), np.imag(dpdt)))
        
    # #     return dydt 


    # def run_qhd(self, t0, tf, n_steps):
    #     # P, K, V, G, ob, g = self.fdm_discretization(self.f, self.grad, self.lb, self.rb, self.N)
    #     P, K, V, G, ob, g = self.dft_discretization(self.f, self.grad, self.lb, self.rb, self.N)

    #     def Ham(t):
    #         a1 = (1 / t**3)
    #         a4 = t**3

    #         H1 = K
    #         H4 = V 
    #         H = csc_matrix(a1 * H1 + a4 * H4) 
    #         return H

    #     # t0 = 1.5 * np.sqrt(self.s)
    #     psi = self.construct_init_state(self.x_data, self.init_state)
    #     snapshot_times = np.linspace(t0, tf, n_steps)
    #     probability = np.zeros((n_steps, self.N))
    #     probability[0] = np.abs(psi)**2
        
    #     for i in range(n_steps-1):
    #         dt = snapshot_times[i+1] - snapshot_times[i]
    #         H = Ham(snapshot_times[i])
    #         psi = expm_multiply(-1j*dt*H, psi)
    #         probability[i+1] = np.abs(psi)**2
        
    #     return snapshot_times, probability


    # def simulator(self, t0, tf, n_steps):
    #     P, K, V, G, ob, g = self.fdm_discretization(self.f, self.grad, self.lb, self.rb, self.N)

    #     def Ham(t):
    #         a1 = (1 / t**3)
    #         a2 = 0.5 * self.beta 
    #         a3 = self.beta * (self.beta + np.sqrt(self.s)) * t**3
    #         a4 = t**3 + 1.5 * (2*self.beta + np.sqrt(self.s)) * t**2

    #         H1 = K
    #         H2 = 0.5 * (G @ P + P @ G)
    #         H3 = np.diag(g**2)
    #         H4 = V 
    #         H = csc_matrix(a1 * H1 + a2 * H2 + a3 * H3 + a4 * H4) 
    #         return H

    #     # t0 = 1.5 * np.sqrt(self.s)
    #     # t_span = [t0, tf]
    #     psi = self.construct_init_state(self.x_data, self.init_state)
    #     snapshot_times = np.linspace(t0, tf, n_steps)
    #     probability = np.zeros((n_steps, self.N))
    #     probability[0] = np.abs(psi)**2
        
    #     for i in range(n_steps-1):
    #         dt = snapshot_times[i+1] - snapshot_times[i]
    #         H = Ham(snapshot_times[i])
    #         psi = expm_multiply(-1j*dt*H, psi)
    #         probability[i+1] = np.abs(psi)**2
        
    #     return snapshot_times, probability


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
    

    # def compute_spectral_gap_qhd(self, t0, tf, n_steps):
    #     snapshot_times = np.linspace(t0, tf, n_steps)
    #     P, K, V, G, ob, g = self.fdm_discretization(self.f, self.grad, self.lb, self.rb, self.N)

    #     def Ham(t):
    #         a1 = (1 / t**3)
    #         a4 = t**3

    #         H1 = K
    #         H4 = V 
    #         H = a1 * H1 + a4 * H4
    #         return H
        
    #     spectral_gap = np.zeros_like(snapshot_times)
    #     for i in range(n_steps):
    #         t_point = snapshot_times[i]
    #         eigenvalues, _ = eigh(Ham(t_point))
    #         spectral_gap[i] = eigenvalues[1] - eigenvalues[0]

    #     return snapshot_times, spectral_gap


    # def compute_spectral_gap_highres_qhd(self, t0, tf, n_steps):
    #     snapshot_times = np.linspace(t0, tf, n_steps)
    #     P, K, V, G, ob, g = self.fdm_discretization(self.f, self.grad, self.lb, self.rb, self.N)

    #     def Ham(t):
    #         a1 = (1 / t**3)
    #         a2 = 0.5 * self.beta 
    #         a3 = self.beta * (self.beta + np.sqrt(self.s)) * t**3
    #         a4 = t**3 + 1.5 * (2*self.beta + np.sqrt(self.s)) * t**2

    #         H1 = K
    #         H2 = 0.5 * (G @ P + P @ G)
    #         H3 = np.diag(g**2)
    #         H4 = V 
    #         H = a1 * H1 + a2 * H2 + a3 * H3 + a4 * H4
    #         return H
        
    #     spectral_gap = np.zeros_like(snapshot_times)
    #     for i in range(n_steps):
    #         t_point = snapshot_times[i]
    #         eigenvalues, _ = eigh(Ham(t_point))
    #         spectral_gap[i] = eigenvalues[1] - eigenvalues[0]

    #     return snapshot_times, spectral_gap




if __name__ == '__main__':
    f = lambda x: (4 * x - 2)**4 - (4 * x - 2 - 1/8)**2
    grad = lambda x: 16 * (4 * x - 2)**3 - 8 * (4 * x - 2 - 1/8)

    lb = 0
    rb = 1
    s = 100
    beta = 2e-3 * np.sqrt(s)
    N = 128

    model = HighResQHD(f, grad, lb, rb, s, beta, N)
    snapshot_times, obj_val = model.qhd_simulator(10, 5e4, 1000)
    
    # plt.plot(model.x_data, probability[-1])
    # plt.show()

