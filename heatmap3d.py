import numpy as np
import time
import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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

        animation.TimedAnimation.__init__(
            self, self.fig, interval=50, blit=True, repeat=False)

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
        # self.ax2 = self.fig.add_subplot(122)

        self.ax1.set_xlim(0, self.OCEAN_SIZE)
        self.ax1.set_ylim(0, self.OCEAN_SIZE)
        self.ax1.set_zlim(0, self.OCEAN_SIZE)
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')

        self.particles_plot = self.ax1.scatter([], [], [], c='b', marker='.')
        self.cameras_plot = self.ax1.scatter([], [], [], c='r', marker='^')
        self.camera_lines = [self.ax1.plot(
            [], [], [], 'r-')[0] for _ in range(self.CAMERAS)]

        # # Initialize heatmap with placeholder data
        # initial_heatmap = np.zeros((50, 50))
        # self.heatmap = self.ax2.imshow(
        #     initial_heatmap, cmap='hot', interpolation='nearest', aspect='auto')
        # self.ax2.set_title('Particle Density Heatmap')
        # self.fig.colorbar(self.heatmap, ax=self.ax2)

    def _draw_frame(self, t):
        # Update particle positions
        self.r += self.dt * self.v

        # Boundary conditions (wrap around)
        self.r %= self.OCEAN_SIZE

        # Update plots
        self.particles_plot._offsets3d = (
            self.r[:, 0], self.r[:, 1], self.r[:, 2])
        self.cameras_plot._offsets3d = (
            self.cam_pos[:, 0], self.cam_pos[:, 1], self.cam_pos[:, 2])

        for i, line in enumerate(self.camera_lines):
            start = self.cam_pos[i]
            # Line length representing camera direction
            end = start + self.cam_dir[i] * 0.5
            line.set_data_3d([start[0], end[0]], [
                             start[1], end[1]], [start[2], end[2]])

        # # Create heatmap
        # heatmap_data, _ = np.histogramdd(self.r[:, :2], bins=(
        #     50, 50), range=((0, self.OCEAN_SIZE), (0, self.OCEAN_SIZE)))

        # # Normalize the heatmap data
        # heatmap_data = heatmap_data / np.max(heatmap_data)

        # self.heatmap.set_array(heatmap_data)
        # self.heatmap.set_clim(0, 1)  # Set color limits

        # Save particle positions
        with open(f'ocean_particles_{time.time()}.csv', 'a', newline='') as file:
            np.savetxt(file, np.round(self.r, 4),
                       delimiter=',', fmt='%.4f', newline='\n')

    def new_frame_seq(self):
        return iter(np.linspace(0, self.max_time, self.Nt))


class OceanSimulation3DHeatmap(OceanSimulation):
    def init_figures(self):
        super().init_figures()
        self.ax3 = self.fig.add_subplot(133, projection='3d')
        self.ax3.set_xlim(0, self.OCEAN_SIZE)
        self.ax3.set_ylim(0, self.OCEAN_SIZE)
        self.ax3.set_zlim(0, self.OCEAN_SIZE)
        self.ax3.set_xlabel('X')
        self.ax3.set_ylabel('Y')
        self.ax3.set_zlabel('Z')
        self.ax3.set_title('3D Particle Density')

        # Initialize voxel plot (placeholder)
        self.voxels = np.zeros((10, 10, 10), dtype=bool)
        self.colors = np.zeros((10, 10, 10), dtype=float)

    def _draw_frame(self, t):
        super()._draw_frame(t)

        # Create 3D density histogram
        bins = (10, 10, 10)
        heatmap_3d, edges = np.histogramdd(self.r, bins=bins, range=[
                                           (0, self.OCEAN_SIZE)] * 3)

        # Normalize the heatmap data
        heatmap_3d = heatmap_3d / np.max(heatmap_3d)

        # Prepare voxel grid and colors
        self.voxels = heatmap_3d > 0.05  # Threshold for visualization
        self.colors = heatmap_3d / heatmap_3d.max()

        # Clear and re-plot voxels
        self.ax3.clear()
        self.ax3.set_xlim(0, self.OCEAN_SIZE)
        self.ax3.set_ylim(0, self.OCEAN_SIZE)
        self.ax3.set_zlim(0, self.OCEAN_SIZE)
        self.ax3.set_xlabel('X')
        self.ax3.set_ylabel('Y')
        self.ax3.set_zlabel('Z')

        self.ax3.voxels(self.voxels, facecolors=plt.cm.hot(
            self.colors), edgecolor='k')


PARTICLES = 100
CAMERAS = 5
OCEAN_SIZE = 10
MAX_TIME = 100
# Run the simulation with 3D heatmap
sim_3d = OceanSimulation3DHeatmap(PARTICLES, CAMERAS, OCEAN_SIZE, MAX_TIME)
plt.show()
