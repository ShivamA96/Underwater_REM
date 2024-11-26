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
    Advanced simulation of particles in a water body with improved boundary handling
    and multiple camera views.

    Key features:
    - Realistic water body boundary interactions
    - Energy-conserving collisions
    - Multiple camera field of view tracking
    - Encounter and velocity tracking
    """

    def __init__(self, n_particles, mass, rad, T, water_dimensions, max_time,
                 camera_fovs=None, dt=0.1):
        """
        Initialize the water body particle simulation.

        Args:
            n_particles (int): Number of particles in the system
            mass (float): Mass of each particle
            rad (float): Particle radius
            T (float): System temperature
            water_dimensions (tuple): Dimensions of the water body (x, y, z)
            max_time (float): Maximum simulation time
            camera_fovs (list): List of camera field of view dictionaries
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

        # Camera Field of Views
        self.camera_fovs = camera_fovs or [
            {"name": "Front", "plane": "XY",
                "slice_pos": self.WATER_Z/2, "thickness": 0.1},
            {"name": "Side", "plane": "XZ",
                "slice_pos": self.WATER_Y/2, "thickness": 0.1},
            {"name": "Top", "plane": "YZ",
                "slice_pos": self.WATER_X/2, "thickness": 0.1}
        ]

        # Tracking encounters and velocities
        self.encounters_per_camera = {
            cam['name']: 0 for cam in self.camera_fovs}
        self.total_encounters = 0
        self.final_avg_velocity = 0

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

    def init_figures(self):
        """
        Initialize the simulation visualization with multiple plots including multiple camera views.
        """
        self.fig = plt.figure(figsize=(20, 15))

        # 3D plot grid layout
        self.ax1 = plt.subplot2grid(
            (3, 4), (0, 0), colspan=2, rowspan=2, projection='3d')
        self.ax2 = plt.subplot2grid((3, 4), (2, 0))  # XY projection
        self.ax5 = plt.subplot2grid((3, 4), (0, 2))  # Velocity distribution

        # Camera view subplots
        self.camera_axes = {}
        for i, cam in enumerate(self.camera_fovs):
            ax = plt.subplot2grid((3, 4), (0, 3) if i ==
                                  0 else (1, 3) if i == 1 else (2, 3))
            ax.set_title(f"{cam['name']} View")
            ax.set_xlabel(f"{cam['plane'][0]} axis")
            ax.set_ylabel(f"{cam['plane'][1]} axis")

            if cam['plane'] == 'XY':
                ax.set_xlim(0, self.WATER_X)
                ax.set_ylim(0, self.WATER_Y)
            elif cam['plane'] == 'XZ':
                ax.set_xlim(0, self.WATER_X)
                ax.set_ylim(0, self.WATER_Z)
            else:  # YZ
                ax.set_xlim(0, self.WATER_Y)
                ax.set_ylim(0, self.WATER_Z)

            plot_line = ax.plot([], [], ls='None', marker='.')[0]
            self.camera_axes[cam['name']] = {
                'ax': ax,
                'line': plot_line
            }

        # Rest of the plot setup remains the same as before
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
        for cam in self.camera_fovs:
            self._drawn_artists.append(self.camera_axes[cam['name']]['line'])

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

    def track_camera_encounters(self):
        """
        Track particle encounters in each camera's field of view.
        An encounter is defined as a particle within the camera's slice.
        """
        for cam in self.camera_fovs:
            plane = cam['plane']
            slice_pos = cam['slice_pos']
            thickness = cam['thickness']

            if plane == 'XY':
                camera_mask = (slice_pos - thickness/2 <= self.r[:, 2]) & \
                              (self.r[:, 2] <= slice_pos + thickness/2)
                cam_view_data_x, cam_view_data_y = self.r[camera_mask,
                                                          0], self.r[camera_mask, 1]
            elif plane == 'XZ':
                camera_mask = (slice_pos - thickness/2 <= self.r[:, 1]) & \
                              (self.r[:, 1] <= slice_pos + thickness/2)
                cam_view_data_x, cam_view_data_y = self.r[camera_mask,
                                                          0], self.r[camera_mask, 2]
            else:  # YZ
                camera_mask = (slice_pos - thickness/2 <= self.r[:, 0]) & \
                              (self.r[:, 0] <= slice_pos + thickness/2)
                cam_view_data_x, cam_view_data_y = self.r[camera_mask,
                                                          1], self.r[camera_mask, 2]

            # Update encounters for this camera
            new_encounters = np.count_nonzero(camera_mask)
            self.encounters_per_camera[cam['name']] = new_encounters

            # Update camera view plot
            cam_plot = self.camera_axes[cam['name']]['line']
            cam_plot.set_data(cam_view_data_x, cam_view_data_y)

    def _draw_frame(self, frameNum):
        """
        Update particle positions and handle interactions for each frame.
        """
        # Update particle positions
        self.r += self.dt * self.v

        # Handle particle-particle and boundary collisions
        self.handle_inter_particle_collisions()
        self.handle_boundary_collisions()

        # Track camera encounters
        self.track_camera_encounters()

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

    def _on_finish(self):
        """
        Compute final statistics when simulation completes.
        """
        # Compute total encounters
        self.total_encounters = sum(self.encounters_per_camera.values())

        # Compute average velocity
        v_mod = np.sqrt(mod(self.v))
        self.final_avg_velocity = np.mean(v_mod)

        # Print results
        print("\nSimulation Completed!")
        print("\nEncounters per Camera:")
        for cam, encounters in self.encounters_per_camera.items():
            print(f"{cam} Camera: {encounters} encounters")

        print(f"\nTotal Encounters: {self.total_encounters}")
        print(f"Average Particle Velocity: {self.final_avg_velocity:.4f} m/s")


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

    # Custom camera views (optional)
    CAMERA_FOVS = [
        {"name": "Front", "plane": "XY", "slice_pos": 0.5, "thickness": 0.1},
        {"name": "Side", "plane": "XZ", "slice_pos": 0.5, "thickness": 0.1},
        {"name": "Top", "plane": "YZ", "slice_pos": 0.5, "thickness": 0.1}
    ]

    # Create and run simulation
    simulation = WaterBodySimulation(
        PARTICLES,
        MASS,
        RADIUS,
        TEMPERATURE,
        WATER_DIMENSIONS,
        MAX_TIME,
        CAMERA_FOVS
    )

    plt.show()


if __name__ == "__main__":
    main()
