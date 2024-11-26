import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import csv
import time

class OceanSimulation(animation.TimedAnimation):
    def __init__(self, n_particles, n_cameras, ocean_size, max_time, dt=0.1):
        self.PART = n_particles
        self.CAMERAS = n_cameras
        self.OCEAN_SIZE = ocean_size
        self.max_time = max_time
        self.dt = dt
        self.Nt = int(max_time / self.dt)

        self.init_particles()
        self.init_cameras()
        self.init_figures()

        animation.TimedAnimation.__init__(self, self.fig, interval=50, blit=True, repeat=False)

    def init_particles(self):
        self.r = np.random.rand(self.PART, 3) * self.OCEAN_SIZE
        self.v = np.random.randn(self.PART, 3) * 0.1  # Random velocities

    def init_cameras(self):
        self.cam_pos = np.random.rand(self.CAMERAS, 3) * self.OCEAN_SIZE
        self.cam_dir = np.random.randn(self.CAMERAS, 3)
        self.cam_dir /= np.linalg.norm(self.cam_dir, axis=1)[:, np.newaxis]

    def init_figures(self):
        self.fig = plt.figure(figsize=(15, 10))
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax2 = self.fig.add_subplot(122)

        self.ax1.set_xlim(0, self.OCEAN_SIZE)
        self.ax1.set_ylim(0, self.OCEAN_SIZE)
        self.ax1.set_zlim(0, self.OCEAN_SIZE)
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')

        self.particles_plot = self.ax1.scatter([], [], [], c='b', marker='.')
        self.cameras_plot = self.ax1.scatter([], [], [], c='r', marker='^')
        self.camera_lines = [self.ax1.plot([], [], [], 'r-')[0] for _ in range(self.CAMERAS)]

        # Initialize heatmap with placeholder data
        initial_heatmap = np.zeros((50, 50))
        self.heatmap = self.ax2.imshow(initial_heatmap, cmap='hot', interpolation='nearest', aspect='auto')
        self.ax2.set_title('Particle Density Heatmap')
        self.fig.colorbar(self.heatmap, ax=self.ax2)

        self._drawn_artists = [self.particles_plot, self.cameras_plot, self.heatmap] + self.camera_lines

    def _draw_frame(self, t):
        # Update particle positions
        self.r += self.dt * self.v
        
        # Boundary conditions (wrap around)
        self.r %= self.OCEAN_SIZE

        # Update plots
        self.particles_plot._offsets3d = (self.r[:, 0], self.r[:, 1], self.r[:, 2])
        self.cameras_plot._offsets3d = (self.cam_pos[:, 0], self.cam_pos[:, 1], self.cam_pos[:, 2])

        for i, line in enumerate(self.camera_lines):
            start = self.cam_pos[i]
            end = start + self.cam_dir[i] * 0.5  # Line length representing camera direction
            line.set_data_3d([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])

        # Create heatmap
        heatmap_data, _ = np.histogramdd(self.r[:, :2], bins=(50, 50), range=((0, self.OCEAN_SIZE), (0, self.OCEAN_SIZE)))
        
        # Normalize the heatmap data
        heatmap_data = heatmap_data / np.max(heatmap_data)
        
        self.heatmap.set_array(heatmap_data)
        self.heatmap.set_clim(0, 1)  # Set color limits

        # Save particle positions
        with open(f'ocean_particles_{time.time()}.csv', 'a', newline='') as file:
            np.savetxt(file, np.round(self.r, 4), delimiter=',', fmt='%.4f', newline='\n')

    def new_frame_seq(self):
        return iter(np.linspace(0, self.max_time, self.Nt))

# Run the simulation
PARTICLES = 1000
CAMERAS = 5
OCEAN_SIZE = 10
MAX_TIME = 100

sim = OceanSimulation(PARTICLES, CAMERAS, OCEAN_SIZE, MAX_TIME)
plt.show()