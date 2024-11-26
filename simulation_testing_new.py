import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import types
from mpl_toolkits.mplot3d import Axes3D
import time
import math

# Physical constants
k_B = 1.380648e-23  # Boltzmann constant (J/K)


def mod(v):
    """
    Compute the squared sum over the last axis of the numpy array.

    Args:
        v (numpy.ndarray): Input array of velocities or positions

    Returns:
        numpy.ndarray: Squared magnitude of vectors
    """
    return np.sum(v * v, axis=-1)


def maxwell_boltzmann_distribution(v, T, m):
    """
    Maxwell-Boltzmann's distribution of probability for velocity magnitude.

    Args:
        v (float): Velocity magnitude
        T (float): Temperature
        m (float): Particle mass

    Returns:
        float: Probability density
    """
    return 4 * np.pi * v**2 * np.power(m / (2 * np.pi * k_B * T), 3 / 2) * np.exp(- m * v**2 / (2 * k_B * T))


class WaterBodySimulation(animation.TimedAnimation):
    """
    Advanced simulation of particles in a water body with improved boundary handling.

    Key features:
    - Realistic water body boundary interactions
    - Energy-conserving collisions
    - Detailed tracking of particle trajectories
    """

    def __init__(self, n_particles, mass, rad, T, water_dimensions, max_time, dt=0.1):
        """
        Initialize the water body particle simulation.

        Args:
            n_particles (int): Number of particles in the system
            mass (float): Mass of each particle
            rad (float): Particle radius
            T (float): System temperature
            water_dimensions (tuple): Dimensions of the water body (x, y, z)
            max_time (float): Maximum simulation time
            dt (float): Time step for simulation
        """
        self.PART = n_particles
        self.MASS = mass
        self.RAD = rad
        self.DIAM = 2 * rad
        self.T = T

        # Water body dimensions
        self.WATER_X, self.WATER_Y, self.WATER_Z = water_dimensions

        self.max_time = max_time
        self.dt = dt
        self.Nt = int(max_time / self.dt)

        # Velocities histogram
        self.min_v = 0
        self.max_v = 3 * np.sqrt(2 * k_B * self.T / self.MASS)
        self.dv = 0.2  # Velocity bin width
        self.Nv = int((self.max_v - self.min_v) / self.dv)

        # Initialize system properties
        self.evaluate_properties()
        self.init_particles()
        self.init_figures()

        animation.TimedAnimation.__init__(
            self, self.fig, interval=1, blit=True, repeat=False)

    def evaluate_properties(self):
        """
        Calculate initial thermodynamic properties of the system.
        """
        water_volume = self.WATER_X * self.WATER_Y * self.WATER_Z
        self.P = self.PART * k_B * self.T / water_volume
        self.U = 1.5 * self.PART * k_B * self.T
        self.vrms = np.sqrt(3 * k_B * self.T / self.MASS)
        self.vmax = np.sqrt(2 * k_B * self.T / self.MASS)
        self.vmed = np.sqrt(8 * k_B * self.T / (np.pi * self.MASS))

        # Velocity histogram initialization
        self.vel_x = np.linspace(self.min_v, self.max_v, self.Nv)
        self.vel_y = np.zeros(self.Nv)

    def init_particles(self):
        """
        Initialize particle positions and velocities with water body constraints.
        """
        # Randomly distribute particles within water body
        self.r = np.random.rand(self.PART, 3) * \
            [self.WATER_X, self.WATER_Y, self.WATER_Z]

        # Generate velocities using Maxwell-Boltzmann distribution
        v_polar = np.random.random((self.PART, 2))

        self.v = np.zeros((self.PART, 3))
        self.v[:, 0] = np.sin(v_polar[:, 0] * np.pi) * \
            np.cos(v_polar[:, 1] * 2 * np.pi)
        self.v[:, 1] = np.sin(v_polar[:, 0] * np.pi) * \
            np.sin(v_polar[:, 1] * 2 * np.pi)
        self.v[:, 2] = np.cos(v_polar[:, 0] * np.pi)

        self.v *= self.vrms

    def init_figures(self):
        """
        Initialize the simulation visualization with multiple plots.
        """
        self.fig = plt.figure(figsize=(15, 10))

        # 3D plot grid layout
        self.ax1 = plt.subplot2grid(
            (3, 3), (0, 0), colspan=2, rowspan=2, projection='3d')
        self.ax2 = plt.subplot2grid((3, 3), (2, 0))  # XY projection
        self.ax5 = plt.subplot2grid((3, 3), (0, 2))  # Velocity distribution

        # 3D plot setup
        box_limits = [0, max(self.WATER_X, self.WATER_Y, self.WATER_Z)]
        self.ax1.set_xlim3d(box_limits)
        self.ax1.set_xlabel('X')
        self.ax1.set_ylim3d(box_limits)
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlim3d(box_limits)
        self.ax1.set_zlabel('Z')

        # Particle plot in 3D
        self.line_3d = self.ax1.plot([], [], [], ls='None', marker='.')[0]

        # XY projection setup
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        self.ax2.set_xlim(0, self.WATER_X)
        self.ax2.set_ylim(0, self.WATER_Y)
        self.line_xy = self.ax2.plot([], [], ls='None', marker='.')[0]

        # Theoretical velocity distribution
        vs = np.linspace(0, self.max_v, 50)
        self.ax5.set_xlabel('Velocity (m/s)')
        self.ax5.set_ylabel('Particle Count')
        self.ax5.plot(vs, self.PART * maxwell_boltzmann_distribution(vs,
                      self.T, self.MASS) * self.dv, color='r')

        # Actual velocity distribution line
        self.line_vel = self.ax5.plot([], [], color='b', lw=0.5)[0]

        self._drawn_artists = [self.line_3d, self.line_xy, self.line_vel]

    def handle_boundary_collisions(self):
        """
        Advanced boundary collision handling with energy conservation.
        """
        boundary_collisions = np.zeros_like(self.r, dtype=bool)

        # Boundary checks
        x_low_collision = self.r[:, 0] < 0
        x_high_collision = self.r[:, 0] > self.WATER_X
        boundary_collisions[:, 0] = x_low_collision | x_high_collision

        y_low_collision = self.r[:, 1] < 0
        y_high_collision = self.r[:, 1] > self.WATER_Y
        boundary_collisions[:, 1] = y_low_collision | y_high_collision

        z_low_collision = self.r[:, 2] < 0
        z_high_collision = self.r[:, 2] > self.WATER_Z
        boundary_collisions[:, 2] = z_low_collision | z_high_collision

        # Coefficient of restitution
        RESTITUTION = 0.95

        # Reflect velocities and adjust positions
        for dim in range(3):
            reflection_indices = boundary_collisions[:, dim]

            # Reverse velocity with energy loss
            self.v[reflection_indices, dim] *= -RESTITUTION

            # Correct particle positions
            if dim == 0:  # X boundary
                self.r[x_low_collision, dim] = -self.r[x_low_collision, dim]
                self.r[x_high_collision, dim] = 2 * \
                    self.WATER_X - self.r[x_high_collision, dim]
            elif dim == 1:  # Y boundary
                self.r[y_low_collision, dim] = -self.r[y_low_collision, dim]
                self.r[y_high_collision, dim] = 2 * \
                    self.WATER_Y - self.r[y_high_collision, dim]
            else:  # Z boundary
                self.r[z_low_collision, dim] = -self.r[z_low_collision, dim]
                self.r[z_high_collision, dim] = 2 * \
                    self.WATER_Z - self.r[z_high_collision, dim]

    def handle_inter_particle_collisions(self):
        """
        Advanced inter-particle collision detection and resolution.
        """
        # Compute pairwise distances
        dists = np.sqrt(
            mod(self.r[:, np.newaxis, :] - self.r[np.newaxis, :, :]))

        # Identify colliding particle pairs
        collision_mask = (dists > 0) & (dists < self.DIAM)
        collision_indices = np.argwhere(collision_mask)

        for i, j in collision_indices:
            if i >= j:  # Avoid duplicate processing
                continue

            # Compute relative position and velocity
            r_ij = self.r[i] - self.r[j]
            v_ij = self.v[i] - self.v[j]

            # Elastic collision resolution
            dot_product = np.dot(v_ij, r_ij)
            distance_squared = np.sum(r_ij ** 2)

            collision_factor = 2 * self.MASS * dot_product / \
                (distance_squared * (2 * self.MASS))

            # Update velocities
            self.v[i] -= collision_factor * r_ij
            self.v[j] += collision_factor * r_ij

            # Slightly separate colliding particles
            overlap_correction = (self.DIAM - dists[i, j]) / 2
            self.r[i] += overlap_correction * r_ij / dists[i, j]
            self.r[j] -= overlap_correction * r_ij / dists[i, j]

    def _draw_frame(self, frameNum):
        """
        Update particle positions and handle interactions for each frame.
        """
        # Update particle positions
        self.r += self.dt * self.v

        # Handle particle-particle and boundary collisions
        self.handle_inter_particle_collisions()
        self.handle_boundary_collisions()

        # Update visualization
        self.update_visualization()

    def update_visualization(self):
        """
        Update plot elements for current particle state.
        """
        # 3D particle positions
        self.line_3d.set_data(self.r[:, 0], self.r[:, 1])
        self.line_3d.set_3d_properties(self.r[:, 2])

        # XY projection
        self.line_xy.set_data(self.r[:, 0], self.r[:, 1])

        # Velocity histogram
        v_mod = np.sqrt(mod(self.v))
        for k in range(self.Nv):
            self.vel_y[k] = np.count_nonzero(
                (k*self.dv < v_mod) & (v_mod < (k + 1)*self.dv))

        self.line_vel.set_data(self.vel_x, self.vel_y)

    def new_frame_seq(self):
        """
        Generate frame sequence for animation.
        """
        return iter(np.linspace(0, self.max_time, self.Nt))


def main():
    """
    Main function to set up and run the water body particle simulation.
    """
    # Simulation parameters
    PARTICLES = 100
    MASS = 1.2e-20  # kg
    RADIUS = 0.01   # m
    TEMPERATURE = 500  # K
    WATER_DIMENSIONS = (1.0, 1.0, 1.0)  # 1m x 1m x 1m water body
    MAX_TIME = 100  # seconds

    # Create and run simulation
    simulation = WaterBodySimulation(
        PARTICLES,
        MASS,
        RADIUS,
        TEMPERATURE,
        WATER_DIMENSIONS,
        MAX_TIME
    )

    plt.show()


if __name__ == "__main__":
    main()
