import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import types
from mpl_toolkits.mplot3d import Axes3D
import csv
import time
import math

k_B = 1.380648e-23  # boltzmann constant (J/K)

def mod(v):
    return np.sum(v * v, axis=-1)

def pmod(v, T, m):
    return 4 * np.pi * v**2 * np.power(m / (2 * np.pi * k_B * T), 3 / 2) * np.exp(- m * v**2 / (2 * k_B * T))

class Simulation(animation.TimedAnimation):
    def __init__(self, n_particles, mass, rad, T, V, max_time, dt=0.1, n_cameras=3):
        self.PART = n_particles
        self.MASS = mass
        self.RAD = rad
        self.DIAM = 2 * rad
        self.T = T
        self.n_cameras = n_cameras

        if isinstance(V, types.FunctionType):
            self.V0 = V(0)
            self.V = V
            self.Vconst = False
        else:
            self.V0 = V
            self.V = lambda t: V
            self.Vconst = True

        self.L = np.power(self.V0, 1/3)
        self.halfL = self.L / 2
        self.A = 6 * self.L**2

        self.max_time = max_time
        self.dt = dt
        self.Nt = int(max_time / self.dt)

        self.evaluate_properties()

        self.min_v = 0
        self.max_v = self.vmax * 3
        self.dv = 0.2
        self.Nv = int((self.max_v - self.min_v) / self.dv)

        self.dP = 1
        self.NP = int(max_time / self.dP)

        self.init_particles()
        self.init_cameras()
        self.init_figures()

        animation.TimedAnimation.__init__(self, self.fig, interval=1, blit=True, repeat=False)

    def evaluate_properties(self):
        self.P = self.PART * k_B * self.T / self.V0
        self.U = 1.5 * self.PART * k_B * self.T
        self.vrms = np.sqrt(3 * k_B * self.T / self.MASS)
        self.vmax = np.sqrt(2 * k_B * self.T / self.MASS)
        self.vmed = np.sqrt(8 * k_B * self.T / (np.pi * self.MASS))

    def init_particles(self):
        self.r = np.random.rand(self.PART, 3) * 2 * (self.halfL - self.RAD) - (self.halfL - self.RAD)
        v_polar = np.random.random((self.PART, 2))
        self.v = np.zeros((self.PART, 3))
        self.v[:, 0] = np.sin(v_polar[:, 0] * np.pi) * np.cos(v_polar[:, 1] * 2 * np.pi)
        self.v[:, 1] = np.sin(v_polar[:, 0] * np.pi) * np.sin(v_polar[:, 1] * 2 * np.pi)
        self.v[:, 2] = np.cos(v_polar[:, 0] * np.pi)
        self.v *= self.vrms

    def init_cameras(self):
        self.cam_positions = np.array([
            [self.halfL, 0, 0],
            [-self.halfL, 0, 0],
            [0, self.halfL, 0],
            [0, -self.halfL, 0],
            [0, 0, self.halfL],
            [0, 0, -self.halfL]
        ])[:self.n_cameras]
        
        self.cam_directions = -self.cam_positions / np.linalg.norm(self.cam_positions, axis=1)[:, np.newaxis]

    def init_figures(self):
        self.fig = plt.figure(figsize=(15, 10))
        
        # Main 3D plot
        self.ax1 = self.fig.add_subplot(221, projection='3d')
        self.ax1.set_xlim([-self.halfL, self.halfL])
        self.ax1.set_ylim([-self.halfL, self.halfL])
        self.ax1.set_zlim([-self.halfL, self.halfL])
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.line_3d = self.ax1.plot([], [], [], ls='None', marker='.')[0]
        
        # Camera view plots
        self.camera_axes = []
        for i in range(self.n_cameras):
            ax = self.fig.add_subplot(2, 3, i+4)
            ax.set_xlim([-self.halfL, self.halfL])
            ax.set_ylim([-self.halfL, self.halfL])
            ax.set_title(f'Camera {i+1} View')
            self.camera_axes.append(ax)
        
        self.camera_lines = [ax.plot([], [], ls='None', marker='.')[0] for ax in self.camera_axes]
        
        # Velocity histogram
        self.ax_vel = self.fig.add_subplot(222)
        vs = np.linspace(0, self.vmax * 3, 50)
        self.ax_vel.set_xlabel(r'$v\ (m/s)$')
        self.ax_vel.set_ylabel(r'$N$')
        self.ax_vel.plot(vs, self.PART * pmod(vs, self.T, self.MASS) * self.dv, color='r')
        self.vel_x = np.linspace(self.min_v, self.max_v, self.Nv)
        self.vel_y = np.zeros(self.Nv)
        self.line_vel = self.ax_vel.plot([], [], color='b', lw=0.5)[0]
        
        # Pressure plot
        self.ax_p = self.fig.add_subplot(223)
        self.ax_p.set_xlabel(r'$t\ (s)$')
        self.ax_p.set_ylabel(r'$P\ (Pa)$')
        if self.Vconst:
            pt = self.PART * k_B * self.T / self.V0
            self.ax_p.plot([0, self.max_time], [pt, pt], color='r', lw=0.5)
        else:
            Vx = self.V(np.linspace(0, self.max_time, self.Nt))
            self.ax_p.plot(Vx, self.PART * k_B * self.T / Vx, color='r', lw=0.5)
        
        self.ex_p = 0.0
        self.last_P = -1
        self.P_x = np.zeros(self.NP)
        self.P_y = np.zeros(self.NP)
        self.line_p = self.ax_p.plot([], [], color='b', lw=0.5)[0]
        
        self._drawn_artists = [self.line_3d, self.line_vel, self.line_p] + self.camera_lines

    def update_volume(self, t):
        self.V0 = self.V(t)
        self.L = np.power(self.V0, 1/3)
        self.halfL = self.L / 2
        self.A = 6 * self.L**2
        
        box_limits = [-self.halfL, self.halfL]
        self.ax1.set_xlim3d(box_limits)
        self.ax1.set_ylim3d(box_limits)
        self.ax1.set_zlim3d(box_limits)
        
        for ax in self.camera_axes:
            ax.set_xlim(box_limits)
            ax.set_ylim(box_limits)

    def _draw_frame(self, t):
        self.update_volume(t)
        
        # Update particle positions
        self.r += self.dt * self.v
        
        # Check for collisions with other particles
        dists = np.sqrt(mod(self.r - self.r[:, np.newaxis]))
        cols2 = (0 < dists) & (dists < self.DIAM)
        idx_i, idx_j = np.nonzero(cols2)
        for i, j in zip(idx_i, idx_j):
            if j < i:
                continue
            rij = self.r[i] - self.r[j]
            d = mod(rij)
            vij = self.v[i] - self.v[j]
            dv = np.dot(vij, rij) * rij / d
            self.v[i] -= dv
            self.v[j] += dv
            self.r[i] += self.dt * self.v[i]
            self.r[j] += self.dt * self.v[j]
        
        # Check for collisions with walls
        walls = np.nonzero(np.abs(self.r) + self.RAD > self.halfL)
        self.v[walls] *= -1
        self.r[walls] -= self.RAD * np.sign(self.r[walls])
        
        # Update main 3D plot
        self.line_3d.set_data(self.r[:, 0], self.r[:, 1])
        self.line_3d.set_3d_properties(self.r[:, 2])
        
        # Update camera views
        for i, (line, cam_pos, cam_dir) in enumerate(zip(self.camera_lines, self.cam_positions, self.cam_directions)):
            # Project particles onto camera plane
            r_rel = self.r - cam_pos
            dist = np.dot(r_rel, cam_dir)
            r_proj = r_rel - dist[:, np.newaxis] * cam_dir
            
            # Calculate 2D coordinates in camera plane
            up = np.array([0, 0, 1])
            right = np.cross(cam_dir, up)
            up = np.cross(right, cam_dir)
            
            x = np.dot(r_proj, right)
            y = np.dot(r_proj, up)
            
            line.set_data(x, y)
        
        # Update velocity histogram
        v_mod = np.sqrt(mod(self.v))
        for k in range(self.Nv):
            self.vel_y[k] = np.count_nonzero((k*self.dv < v_mod) & (v_mod < (k + 1)*self.dv))
        self.line_vel.set_data(self.vel_x, self.vel_y)
        
        # Update pressure plot
        self.ex_p += 2 * self.MASS * np.sum(np.abs(self.v[walls]))
        i = int(t / self.dP)
        if i > self.last_P + 1:
            self.last_P = i - 1
            A_avg = self.A if self.Vconst else (self.A + 6 * np.power(self.V(t - self.dP), 2/3)) / 2
            self.P_x[self.last_P] = t
            self.P_y[self.last_P] = self.ex_p / (self.dP * A_avg)
            self.ex_p = 0.0
            self.line_p.set_data(self.P_x[:i], self.P_y[:i])
            self.ax_p.set_ylim(np.min(self.P_y[:i]), np.max(self.P_y[:i]))

    def new_frame_seq(self):
        return iter(np.linspace(0, self.max_time, self.Nt))

    def save_data(self):
        with open('pressure.txt', 'w') as outf:
            t = np.linspace(0, self.max_time, self.NP)
            for i in range(self.NP):
                outf.write('%.5f\t%.5f\t%.5g\n' % (t[i], self.P_x[i], self.P_y[i]))
        
        with open('hist_vel.txt', 'w') as outf:
            for i in range(self.Nv):
                outf.write('%.5f\t%.5g\n' % (self.vel_x[i], self.vel_y[i]))

# Simulation parameters
PARTICLES = 100
MASS = 1.2e-20
RADIUS = 0.01
TEMPERATURE = 500
VOLUME = 10
T_MAX = 1000
N_CAMERAS = 3

# Run simulation
ani = Simulation(PARTICLES, MASS, RADIUS, TEMPERATURE, VOLUME, T_MAX, 0.1, N_CAMERAS)
plt.show()
ani.save_data()