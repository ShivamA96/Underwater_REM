import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import csv
import time
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation"""
    num_particles: int
    mass: float
    radius: float
    temperature: float
    volume: float
    max_time: float
    dt: float = 0.1
    k_B: float = 1.380648e-23  # Boltzmann constant (J/K)

class EnhancedParticleSimulation:
    """
    Enhanced particle simulation combining ideal gas dynamics with camera tracking.
    Supports multiple observation methods and data collection.
    """
    
    def __init__(self, config: SimulationConfig, camera_positions=None):
        self.config = config
        
        # Set camera positions first
        if camera_positions is None:
            self.camera_positions = np.array([[0.3, 0.3, 0.3]])
        else:
            self.camera_positions = camera_positions
            
        # Initialize camera encounters
        self.camera_encounters = {i: {'counts': {}, 'visible': set()} 
                                for i in range(len(self.camera_positions))}
        
        # Then proceed with other setup
        self.setup_simulation_parameters()
        self.initialize_particles()
        self.setup_visualization()
        self.setup_data_collection()
    
    def setup_simulation_parameters(self):
        """Initialize physical parameters of the simulation"""
        self.L = np.power(self.config.volume, 1/3)
        self.halfL = self.L / 2
        self.area = 6 * self.L**2
        
        # Calculate thermodynamic properties
        self.pressure = (self.config.num_particles * self.config.k_B * 
                        self.config.temperature / self.config.volume)
        self.energy = 1.5 * self.config.num_particles * self.config.k_B * self.config.temperature
        self.v_rms = np.sqrt(3 * self.config.k_B * self.config.temperature / self.config.mass)
        
    def initialize_particles(self):
        """Initialize particle positions and velocities"""
        # Random positions within bounds
        self.positions = (np.random.rand(self.config.num_particles, 3) * 
                         2 * (self.halfL - self.config.radius) - 
                         (self.halfL - self.config.radius))
        
        # Velocities from Maxwell-Boltzmann distribution
        angles = np.random.random((self.config.num_particles, 2))
        self.velocities = np.zeros((self.config.num_particles, 3))
        
        # Convert spherical to Cartesian coordinates for velocities
        self.velocities[:, 0] = np.sin(angles[:, 0] * np.pi) * np.cos(angles[:, 1] * 2 * np.pi)
        self.velocities[:, 1] = np.sin(angles[:, 0] * np.pi) * np.sin(angles[:, 1] * 2 * np.pi)
        self.velocities[:, 2] = np.cos(angles[:, 0] * np.pi)
        
        self.velocities *= self.v_rms
        
        # Particle properties
        self.sizes = np.full(self.config.num_particles, self.config.radius)
        self.colors = plt.cm.viridis(np.linspace(0, 1, self.config.num_particles))
        
    def setup_visualization(self):
        """Setup matplotlib figures and axes"""
        self.fig = plt.figure(figsize=(15, 10))
        
        # Main 3D view
        self.ax_main = self.fig.add_subplot(231, projection='3d')
        self.ax_main.set_title('Main View')
        
        # Camera views
        num_cameras = len(self.camera_positions)
        self.camera_axes = []
        for i in range(num_cameras):
            ax = self.fig.add_subplot(232 + i, projection='3d')
            ax.set_title(f'Camera {i+1} View')
            self.camera_axes.append(ax)
            
        # Velocity distribution
        self.ax_vel = self.fig.add_subplot(235)
        self.ax_vel.set_title('Velocity Distribution')
        
        # Pressure plot
        self.ax_pressure = self.fig.add_subplot(236)
        self.ax_pressure.set_title('Pressure vs Time')
        
    def setup_data_collection(self):
        """Initialize data collection arrays and files"""
        self.time_points = np.arange(0, self.config.max_time, self.config.dt)
        self.pressure_history = []
        self.velocity_history = []
        
        # Create CSV file for position tracking
        timestamp = int(time.time())
        self.data_filename = f'particle_data_{timestamp}.csv'
        with open(self.data_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'particle_id', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
            
    def update(self, frame):
        """Update particle positions and handle collisions"""
        # Update positions
        self.positions += self.velocities * self.config.dt
        
        # Handle particle-particle collisions
        self.handle_collisions()
        
        # Handle wall collisions
        self.handle_wall_collisions()
        
        # Update camera tracking
        self.update_camera_tracking()
        
        # Record data
        self.record_frame_data(frame * self.config.dt)
        
        # Update visualization
        self.update_visualization()
        
        return self.collect_artists()
    
    def handle_collisions(self):
        """Handle particle-particle collisions"""
        distances = np.sqrt(np.sum((self.positions[:, np.newaxis] - self.positions) ** 2, axis=2))
        collisions = (0 < distances) & (distances < 2 * self.config.radius)
        
        for i, j in zip(*np.where(collisions)):
            if j <= i:
                continue
                
            # Elastic collision physics
            r_ij = self.positions[i] - self.positions[j]
            d = np.sum(r_ij ** 2)
            v_ij = self.velocities[i] - self.velocities[j]
            dv = np.dot(v_ij, r_ij) * r_ij / d
            
            self.velocities[i] -= dv
            self.velocities[j] += dv
            
    def handle_wall_collisions(self):
        """Handle collisions with container walls"""
        wall_collisions = np.abs(self.positions) + self.config.radius > self.halfL
        self.velocities[wall_collisions] *= -1
        self.positions[wall_collisions] = np.sign(self.positions[wall_collisions]) * (self.halfL - self.config.radius)
    
    def update_camera_tracking(self):
        """Update particle tracking from each camera position"""
        for i, cam_pos in enumerate(self.camera_positions):
            # Calculate particles in camera FOV
            distances = np.linalg.norm(self.positions - cam_pos, axis=1)
            in_view = distances < self.halfL / 2  # Simple FOV calculation
            
            # Update visible particles for this camera
            newly_visible = set(np.where(in_view)[0]) - self.camera_encounters[i]['visible']
            self.camera_encounters[i]['visible'] = set(np.where(in_view)[0])
            
            # Count new encounters
            for particle_id in newly_visible:
                color_key = tuple(self.colors[particle_id])
                self.camera_encounters[i]['counts'][color_key] = \
                    self.camera_encounters[i]['counts'].get(color_key, 0) + 1
    
    def record_frame_data(self, time):
        """Record position and velocity data to CSV"""
        with open(self.data_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            for i in range(self.config.num_particles):
                writer.writerow([time, i, 
                               *self.positions[i], 
                               *self.velocities[i]])
    
    def update_visualization(self):
        """Update all visualization elements"""
        # Clear previous frame
        self.ax_main.cla()
        for ax in self.camera_axes:
            ax.cla()
            
        # Plot main view
        self.ax_main.scatter(*self.positions.T, c=self.colors, s=self.sizes*1000)
        self.ax_main.set_xlim([-self.halfL, self.halfL])
        self.ax_main.set_ylim([-self.halfL, self.halfL])
        self.ax_main.set_zlim([-self.halfL, self.halfL])
        
        # Plot camera views
        for i, (ax, cam_pos) in enumerate(zip(self.camera_axes, self.camera_positions)):
            visible = self.camera_encounters[i]['visible']
            if visible:
                visible_positions = self.positions[list(visible)]
                visible_colors = self.colors[list(visible)]
                ax.scatter(*visible_positions.T, c=visible_colors, s=self.sizes[list(visible)]*1000)
            ax.set_xlim([-self.halfL, self.halfL])
            ax.set_ylim([-self.halfL, self.halfL])
            ax.set_zlim([-self.halfL, self.halfL])
            
        # Update velocity distribution
        v_magnitudes = np.linalg.norm(self.velocities, axis=1)
        self.ax_vel.hist(v_magnitudes, bins=30, density=True)
        self.ax_vel.set_xlabel('Velocity (m/s)')
        self.ax_vel.set_ylabel('Probability Density')
        
        # Update pressure plot
        instant_pressure = self.calculate_pressure()
        self.pressure_history.append(instant_pressure)
        self.ax_pressure.plot(self.time_points[:len(self.pressure_history)], 
                            self.pressure_history, 'b-')
        self.ax_pressure.set_xlabel('Time (s)')
        self.ax_pressure.set_ylabel('Pressure (Pa)')
    
    def calculate_pressure(self):
        """Calculate instantaneous pressure from wall collisions"""
        wall_collisions = np.abs(self.positions) + self.config.radius > self.halfL
        momentum_transfer = 2 * self.config.mass * np.sum(np.abs(self.velocities[wall_collisions]))
        return momentum_transfer / (self.config.dt * self.area)
    
    def collect_artists(self):
        """Collect all artists for animation"""
        return []  # Animation will redraw everything
    
    def run_simulation(self):
        """Run the complete simulation"""
        anim = animation.FuncAnimation(
            self.fig, self.update,
            frames=len(self.time_points),
            interval=50,
            blit=False
        )
        plt.show()
        return anim

# Example usage
if __name__ == "__main__":
    config = SimulationConfig(
        num_particles=100,
        mass=1.2e-20,
        radius=0.01,
        temperature=500,
        volume=10,
        max_time=10,
        dt=0.1
    )
    
    camera_positions = np.array([
        [0.3, 0.3, 0.3],
        [-0.3, 0.3, 0.3],
        [0.0, -0.3, 0.3]
    ])
    
    sim = EnhancedParticleSimulation(config, camera_positions)
    anim = sim.run_simulation()