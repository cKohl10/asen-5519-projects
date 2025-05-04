# Vehicle Definitions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.common import cont2disc_AQ

class Theta:
    def __init__(self, theta_set=None):
        if theta_set is not None:
            self.set_theta(theta_set)

    def set_element(self, dict, element):
        try:
            return_element = dict[element]
        except:
            return_element = None
        return return_element

    def set_theta(self, theta_set):
        self.A = self.set_element(theta_set, "A")
        self.B = self.set_element(theta_set, "B")
        self.C = self.set_element(theta_set, "C")
        self.Gamma = self.set_element(theta_set, "Gamma")
        self.Sigma = self.set_element(theta_set, "Sigma")
        self.mu0 = self.set_element(theta_set, "mu0")
        self.V0 = self.set_element(theta_set, "V0")
        self.N = self.set_element(theta_set, "N")
        self.Ns = self.set_element(theta_set, "Ns")
        self.Nx = self.set_element(theta_set, "Nx")
        self.Nu = self.set_element(theta_set, "Nu")
        self.Nk = self.set_element(theta_set, "Nk")
    
    def unpack_theta(self):
        return self.A, self.B, self.C, self.Gamma, self.Sigma, self.mu0, self.V0, self.N, self.Ns, self.Nx, self.Nu, self.Nk
    
    def get_theta_set(self):
        theta_set = {}
        for key, value in self.__dict__.items():
            theta_set[key] = value
        return theta_set
    
    def copy(self):
        return Theta(self.get_theta_set())

class Vehicle:
    def __init__(self):
        self.state_history = [] # Z
        self.obs_history = [] # X
        self.time_history = [] # t
        self.control_history = [] # U
        self.current_time = 0
        self.state_names = []
        self.collision_flag = False
        self.vehicle_name = "Vehicle"
        self.fig_path = None

    def reset(self):
        self.state_history = []
        self.time_history = []
        self.control_history = []
        self.obs_history = []
        self.current_time = 0

    def step(self, policy):
        pass

    def prediction_step(self, policy):
        pass

    def set_dynamics(self, theta):
        pass

    def set_A(self, A):
        self.A = A

    def set_B(self, B):
        self.B = B
    
    def set_C(self, C):
        self.C = C

    def animate(self):
        pass
    
    def plot_states(self, title=None, figsize=(12, 8), states_fig=None, controls_fig=None, obs_fig=None):
        """
        Plot all state variables over time.
        
        Parameters:
        ----------
        title : str, optional
            Title for the plot
        figsize : tuple, optional
            Figure size (width, height) in inches
        """
        if len(self.state_history) == 0 or len(self.control_history) == 0:
            print("No history data to plot.")
            return
            
        # Convert history lists to numpy arrays for easier manipulation
        states = np.array(self.state_history)
        obs = np.array(self.obs_history)
        controls = np.array(self.control_history)
        times = np.array(self.time_history)
        
        # Get number of state and control variables
        n_states = states.shape[1]
        n_controls = controls.shape[1]
        n_obs = obs.shape[1]

        # Create figure for states
        if states_fig is None:
            fig_states, axes_states = plt.subplots(n_states, 1, figsize=figsize, sharex=True)
        else:
            fig_states, axes_states = states_fig, states_fig.axes
        if n_states == 1:
            axes_states = [axes_states]
            
        # Plot each state variable
        for i in range(n_states):
            axes_states[i].plot(times, states[:, i])
            axes_states[i].set_ylabel(self.state_names[i])
            axes_states[i].grid(True)
            
        # Add labels and title for states
        axes_states[-1].set_xlabel('Time')
        fig_states.suptitle(f"{self.vehicle_name} States" if not title else f"{self.vehicle_name} States - {title}")

        # Create figure for observations
        if obs_fig is None:
            fig_obs, axes_obs = plt.subplots(n_obs, 1, figsize=figsize, sharex=True)
        else:
            fig_obs, axes_obs = obs_fig, obs_fig.axes
        if n_obs == 1:
            axes_obs = [axes_obs]

        # Plot each observation variable
        for i in range(n_obs):
            axes_obs[i].plot(times, obs[:, i])
            axes_obs[i].set_ylabel(self.state_names[i])
            axes_obs[i].grid(True)

        # Add labels and title for observations
        axes_obs[-1].set_xlabel('Time')
        fig_obs.suptitle(f"{self.vehicle_name} Observations" if not title else f"{self.vehicle_name} Observations - {title}")
            
        plt.tight_layout()
        
        # Create separate figure for controls
        if controls_fig is None:
            fig_controls, axes_controls = plt.subplots(n_controls, 1, figsize=figsize, sharex=True)
        else:
            fig_controls, axes_controls = controls_fig, controls_fig.axes
        if n_controls == 1:
            axes_controls = [axes_controls]
            
        # Plot each control variable with orange lines
        # Use times[1:] because controls are applied BETWEEN states
        for j in range(n_controls):
            axes_controls[j].plot(times[1:], controls[:, j], color='orange')
            axes_controls[j].set_ylabel(f'Control {j+1}')
            axes_controls[j].grid(True)
            
        # Add labels and title for controls
        axes_controls[-1].set_xlabel('Time')
        fig_controls.suptitle("Controls" if not title else f"Controls - {title}")

        plt.tight_layout()

        # Save figures if save_path is provided
        if self.fig_path is not None:
            fig_states.savefig(f"{self.fig_path}_states.png")
            fig_obs.savefig(f"{self.fig_path}_obs.png")
            fig_controls.savefig(f"{self.fig_path}_controls.png")

class SimpleLinearModel(Vehicle):
    def __init__(self, theta, fig_path=None):
        super().__init__()
        self.state_names = []
        self.vehicle_name = "Simple Linear Model"
        self.fig_path = fig_path
        self.set_dynamics(theta)
        self.reset()

    def reset(self, s0=None):
        super().reset()
        self.Ns = self.A.shape[0]
        self.Nx = self.C.shape[0]
        self.Nu = self.B.shape[1]
        self.state_names = [f'x{i+1}' for i in range(self.Ns)]
        self.z0 = self.mu0 + np.random.multivariate_normal(np.zeros(self.Ns), self.V0)
        self.z = self.z0.copy()
        self.state_history = [self.z0.copy()]
        self.obs_history = [self.C @ self.z0]
        self.time_history = [self.current_time]

    def set_dynamics(self, theta):
        self.theta = theta
        self.A, self.B, self.C, self.Gamma, self.Sigma, self.mu0, self.V0, self.N, self.Ns, self.Nx, self.Nu, self.Nk = theta.unpack_theta()

    def get_theta(self):
        parameters = ["A", "B", "C", "Gamma", "Sigma", "mu0", "V0", "N", "Ns", "Nx", "Nu", "Nk"]
        theta_set = {}
        for parameter in parameters:
            theta_set[parameter] = getattr(self, parameter)
        return Theta(theta_set)
    
    def discrete_step(self, policy, dt, t):
        u = policy.get_action(self.z, t)
        self.control_history.append(u.copy())
        self.z = self.A @ self.z + self.B @ u + np.random.multivariate_normal(np.zeros(self.Ns), self.Gamma)
        self.x = self.C @ self.z + np.random.multivariate_normal(np.zeros(self.Nx), self.Sigma)
        self.current_time += dt
        self.state_history.append(self.z.copy())
        self.obs_history.append(self.x.copy())
        self.time_history.append(self.current_time)
        return self.z, self.x
    
    def continuous_step(self, policy, dt, t):
        u = policy.get_action(self.z, t)
        self.control_history.append(u.copy())
        sdot = self.A @ self.z + self.B @ u + np.random.multivariate_normal(np.zeros(self.Ns), self.Gamma)
        self.z = self.z + sdot * dt
        self.x = self.C @ self.z + np.random.multivariate_normal(np.zeros(self.Nx), self.Sigma)
        self.current_time += dt
        self.state_history.append(self.z.copy())
        self.obs_history.append(self.x.copy())
        self.time_history.append(self.current_time)
        return self.z, self.x
    
class MassSpringDamper(SimpleLinearModel):
    def __init__(self, theta, fig_path=None):
        super().__init__(theta, fig_path)
        self.vehicle_name = "Mass-Spring-Damper"
        self.reset()

    def reset(self, s0=None):
        super().reset()

    def p_to_dynamics(self, p):
        self.M1 = p[0]  # mass 1
        self.M2 = p[1]  # mass 2
        self.K1 = p[2]  # spring constant 1
        self.K2 = p[3]  # spring constant 2
        self.B = p[4]  # damping constant
        self.r = p[5]  # radius of the masses
        self.COR = p[6]  # Coefficient of restitution [0,1]

        # Underlying Dynamics
        self.A = np.array([ [0,0,1,0,0,0],
                            [0,0,0,1,0,0],
                            [0,0,-self.B/self.M1, 0, -1/self.M1, 1/self.M1],
                            [0,0, 0, 0, 0, -1/self.M2],
                            [0,0,self.K1, 0, 0, 0],
                            [0,0,-self.K2, self.K2, 0, 0]])
    
        self.B = np.array([ [0,0],
                            [0,0],
                            [0, -self.B/self.M1],
                            [1/self.M2, 0],
                            [0, -self.K1],
                            [0, 0]])
        
    def continuous_step(self, policy, dt, t):
        # Check for collisions with velocity direction consideration
        # x1, x2 = self.z[0], self.z[1]
        # v1, v2 = self.z[2], self.z[3]

        # if np.abs(x1 - x2) < 2*self.r and (v2 - v1) < 0:  # Only collide when approaching
        #     self.collision_flag = True
        #     # Calculate new velocities using physics of collisions
        #     m1, m2 = self.M1, self.M2
        #     e = self.COR
    
        #     # Conservation of momentum with COR
        #     new_v1 = ((m1 - e*m2)*v1 + (1 + e)*m2*v2) / (m1 + m2)
        #     new_v2 = ((1 + e)*m1*v1 + (m2 - e*m1)*v2) / (m1 + m2)
    
        #     self.z[2] = new_v1
        #     self.z[3] = new_v2

        # if x1 <= self.r and v1 <= 0:
        #     self.z[2] = -self.z[2]
        # if x2 <= self.r and v2 <= 0:
        #     self.z[3] = -self.z[3]

        u = policy.get_action(self.z, t)
        self.control_history.append(u.copy())
        sdot = self.A @ self.z + self.B @ u + np.random.multivariate_normal(np.zeros(self.Ns), self.Gamma)
        self.z = self.z + sdot * dt
        self.x = self.C @ self.z + np.random.multivariate_normal(np.zeros(self.Nx), self.Sigma)
        self.current_time += dt
        self.state_history.append(self.z.copy())
        self.obs_history.append(self.x.copy())
        self.time_history.append(self.current_time)
        return self.z, self.x

    
    def animate(self, save_path=None):
        # This problem is two masses connected by springs and dampers.
        # x1 and x2 are the positions of the masses.
        if len(self.state_history) == 0:
            print("No state history to animate.")
            return
            
        # Convert history lists to numpy arrays
        states = np.array(self.obs_history)
        times = np.array(self.time_history)
        
        # Extract position data
        x1_positions = states[:, 0]  # First mass position
        x2_positions = states[:, 1]  # Second mass position
        
        # Determine the range for plotting, accounting for the mass radius
        min_pos = min(np.min(x1_positions), np.min(x2_positions)) - self.r - 1
        max_pos = max(np.max(x1_positions), np.max(x2_positions)) + self.r + 1
        
        # Set up the figure and axis with equal aspect ratio
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_xlim(min_pos, max_pos)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('Position')
        ax.set_title(f'{self.vehicle_name} Animation')
        ax.grid(True)
        
        # Use circles for the masses so that they are centered on the positions.
        from matplotlib.patches import Circle  # Ensure Circle is imported
        mass1 = Circle((x1_positions[0], 0), radius=self.r, fc='blue', ec='black')
        mass2 = Circle((x2_positions[0], 0), radius=self.r, fc='red', ec='black')
        
        # Create spring elements with different colors
        wall_spring = plt.Line2D([0, x1_positions[0]], [0, 0], 
                               color='blue', linewidth=2, linestyle='-')
        mass_spring = plt.Line2D([x1_positions[0], x2_positions[0]], [0, 0], 
                               color='red', linewidth=2, linestyle='-')
        
        # Add wall visualization
        wall = plt.Line2D([0, 0], [-1, 1], color='black', linewidth=4)
        ax.add_line(wall)
        
        # Add elements to the plot (remove left_spring)
        ax.add_patch(mass1)
        ax.add_patch(mass2)
        ax.add_line(wall_spring)
        ax.add_line(mass_spring)
        
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        # Function to initialize the animation, positioning the circles at their centers
        def init():
            mass1.center = (x1_positions[0], 0)
            mass2.center = (x2_positions[0], 0)
            wall_spring.set_data([0, x1_positions[0]], [0, 0])
            mass_spring.set_data([x1_positions[0], x2_positions[0]], [0, 0])
            time_text.set_text(f'Time: {times[0]:.2f} s')
            return mass1, mass2, wall_spring, mass_spring, time_text
        
        # Function to update the animation for each frame, keeping the circles centered.
        def update(frame):
            mass1.center = (x1_positions[frame], 0)
            mass2.center = (x2_positions[frame], 0)
            wall_spring.set_data([0, x1_positions[frame]], [0, 0])
            mass_spring.set_data([x1_positions[frame], x2_positions[frame]], [0, 0])
            time_text.set_text(f'Time: {times[frame]:.2f} s')
            return mass1, mass2, wall_spring, mass_spring, time_text
        
        # Create the animation
        ani = animation.FuncAnimation(
            fig, update, frames=range(len(times)),
            init_func=init, blit=True, interval=50)
        
        # Save the animation if a path is provided
        if save_path:
            ani.save(save_path, writer='pillow', fps=30)
            print(f"Animation saved to {save_path}")
        
        # Display the animation
        plt.show()
        
        return ani  # Still return the animation object for flexibility
        

# class MassSpringDamper(Vehicle):
#     def __init__(self, theta):
#         super().__init__()
#         self.state_names = ['x1', 'x2', 'v1', 'v2', 'Fk1', 'Fk2'] # x1 and x2 are the positions of the masses, v1 and v2 are the velocities of the masses, Fk1 and Fk2 are the forces exerted by the springs
#         self.vehicle_name = "Mass-Spring-Damper"
#         self.r = 0.3 # radius of the masses, can be overwritten by the p_to_dynamics function

#         self.set_dynamics(theta)
#         self.reset()

#     def reset(self, s0=None):
#         super().reset()
#         self.Ns = self.A.shape[0] # Number of states
#         self.Nx = self.C.shape[0] # Number of observable states
#         self.Nu = self.B.shape[1] # Number of controls
#         self.s0 = self.mu0 + np.random.multivariate_normal(np.zeros(self.Ns), self.V0)
#         self.s = self.s0.copy()
#         self.state_history.append(self.C @ self.s0)
#         self.time_history.append(self.current_time)

#     def set_dynamics(self, theta):
#         if "A" in theta:
#             self.A = theta["A"] # State Transition Matrix
#         if "Gamma" in theta:
#             self.Gamma = theta["Gamma"] # State Transition Noise Covariance
#         if "C" in theta:
#             self.C = theta["C"] # Observation Matrix
#         if "Sigma" in theta:
#             self.Sigma = theta["Sigma"] # Observation Noise Covariance
#         if "mu0" in theta:
#             self.mu0 = theta["mu0"] # Initial State Mean
#         if "V0" in theta:
#             self.V0 = theta["V0"] # Initial State Covariance
#         if "B" in theta:
#             self.B = theta["B"] # Control Matrix

#     def get_theta(self, N):
#         return {
#             "A": self.A,
#             "Gamma": self.Gamma,
#             "C": self.C,
#             "Sigma": self.Sigma,
#             "mu0": self.mu0,
#             "V0": self.V0,
#             "B": self.B,
#             "Ns": self.Ns,
#             "Nx": self.Nx,
#             "Nu": self.Nu
#         }

#     def p_to_dynamics(self, p):
#         self.M1 = p[0]  # mass 1
#         self.M2 = p[1]  # mass 2
#         self.K1 = p[2]  # spring constant 1
#         self.K2 = p[3]  # spring constant 2
#         self.B = p[4]  # damping constant
#         self.r = p[5]  # radius of the masses
#         self.COR = p[6]  # Coefficient of restitution [0,1]

#         # Underlying Dynamics
#         self.A = np.array([ [0,0,1,0,0,0],
#                             [0,0,0,1,0,0],
#                             [0,0,-self.B/self.M1, 0, -1/self.M1, 1/self.M1],
#                             [0,0, 0, 0, 0, -1/self.M2],
#                             [0,0,self.K1, 0, 0, 0],
#                             [0,0,-self.K2, self.K2, 0, 0]])
        
#         self.B = np.array([ [0,0],
#                             [0,0],
#                             [0, -self.B/self.M1],
#                             [1/self.M2, 0],
#                             [0, -self.K1],
#                             [0, 0]])

#     def step(self, policy, dt, t):

#         # Check for collisions with velocity direction consideration
#         x1, x2 = self.s[0], self.s[1]
#         v1, v2 = self.s[2], self.s[3]
        
#         if np.abs(x1 - x2) < 2*self.r and (v2 - v1) < 0:  # Only collide when approaching
#             self.collision_flag = True
#             # Calculate new velocities using physics of collisions
#             m1, m2 = self.M1, self.M2
#             e = self.COR
            
#             # Conservation of momentum with COR
#             new_v1 = ((m1 - e*m2)*v1 + (1 + e)*m2*v2) / (m1 + m2)
#             new_v2 = ((1 + e)*m1*v1 + (m2 - e*m1)*v2) / (m1 + m2)
            
#             self.s[2] = new_v1
#             self.s[3] = new_v2

#         if x1 <= self.r and v1 <= 0:
#             self.s[2] = -self.s[2]
#         if x2 <= self.r and v2 <= 0:
#             self.s[3] = -self.s[3]

#         # Get the action from the policy
#         u = policy.get_action(self.s, t)
#         self.control_history.append(u.copy())

#         # Step the dynamics
#         sdot = self.A @ self.s + self.B @ u

#         # This uses the CONTINUOUS time dynamics matrices A and B
#         self.s = self.s + sdot * dt + np.random.multivariate_normal(np.zeros(self.Ns), self.Gamma)
#         self.x = self.C @ self.s + np.random.multivariate_normal(np.zeros(self.Nx), self.Sigma)
        
#         # Update time
#         self.current_time += dt
            
        
#         # Store state and time in history
#         self.state_history.append(self.x.copy())
#         self.time_history.append(self.current_time)

#         # Return the next state
#         return self.s, self.x
    
#     def prediction_step(self, policy, dt, t):

#         # Get the action from the policy
#         u = policy.get_action(self.s, t)
#         self.control_history.append(u.copy())

#         # This uses the DISCRETE time dynamics matrices A and C
#         self.s = self.A @ self.s + np.random.multivariate_normal(np.zeros(self.Ns), self.Gamma)
#         self.x = self.C @ self.s + np.random.multivariate_normal(np.zeros(self.Nx), self.Sigma)

#         # Update time
#         self.current_time += dt
        
#         # Store state and time in history
#         self.state_history.append(self.x.copy())
#         self.time_history.append(self.current_time)
        
#         # Return the next state
#         return self.s, self.x
        
    
#     def animate(self, save_path=None):
#         # This problem is two masses connected by springs and dampers.
#         # x1 and x2 are the positions of the masses.
#         if len(self.state_history) == 0:
#             print("No state history to animate.")
#             return
            
#         # Convert history lists to numpy arrays
#         states = np.array(self.state_history)
#         times = np.array(self.time_history)
        
#         # Extract position data
#         x1_positions = states[:, 0]  # First mass position
#         x2_positions = states[:, 1]  # Second mass position
        
#         # Determine the range for plotting, accounting for the mass radius
#         min_pos = min(np.min(x1_positions), np.min(x2_positions)) - self.r - 1
#         max_pos = max(np.max(x1_positions), np.max(x2_positions)) + self.r + 1
        
#         # Set up the figure and axis with equal aspect ratio
#         fig, ax = plt.subplots(figsize=(10, 4))
#         ax.set_xlim(min_pos, max_pos)
#         ax.set_ylim(-1, 1)
#         ax.set_aspect('equal')
#         ax.set_xlabel('Position')
#         ax.set_title(f'{self.vehicle_name} Animation')
#         ax.grid(True)
        
#         # Use circles for the masses so that they are centered on the positions.
#         from matplotlib.patches import Circle  # Ensure Circle is imported
#         mass1 = Circle((x1_positions[0], 0), radius=self.r, fc='blue', ec='black')
#         mass2 = Circle((x2_positions[0], 0), radius=self.r, fc='red', ec='black')
        
#         # Create spring elements with different colors
#         wall_spring = plt.Line2D([0, x1_positions[0]], [0, 0], 
#                                color='blue', linewidth=2, linestyle='-')
#         mass_spring = plt.Line2D([x1_positions[0], x2_positions[0]], [0, 0], 
#                                color='red', linewidth=2, linestyle='-')
        
#         # Add wall visualization
#         wall = plt.Line2D([0, 0], [-1, 1], color='black', linewidth=4)
#         ax.add_line(wall)
        
#         # Add elements to the plot (remove left_spring)
#         ax.add_patch(mass1)
#         ax.add_patch(mass2)
#         ax.add_line(wall_spring)
#         ax.add_line(mass_spring)
        
#         time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
#         # Function to initialize the animation, positioning the circles at their centers
#         def init():
#             mass1.center = (x1_positions[0], 0)
#             mass2.center = (x2_positions[0], 0)
#             wall_spring.set_data([0, x1_positions[0]], [0, 0])
#             mass_spring.set_data([x1_positions[0], x2_positions[0]], [0, 0])
#             time_text.set_text(f'Time: {times[0]:.2f} s')
#             return mass1, mass2, wall_spring, mass_spring, time_text
        
#         # Function to update the animation for each frame, keeping the circles centered.
#         def update(frame):
#             mass1.center = (x1_positions[frame], 0)
#             mass2.center = (x2_positions[frame], 0)
#             wall_spring.set_data([0, x1_positions[frame]], [0, 0])
#             mass_spring.set_data([x1_positions[frame], x2_positions[frame]], [0, 0])
#             time_text.set_text(f'Time: {times[frame]:.2f} s')
#             return mass1, mass2, wall_spring, mass_spring, time_text
        
#         # Create the animation
#         ani = animation.FuncAnimation(
#             fig, update, frames=range(len(times)),
#             init_func=init, blit=True, interval=50)
        
#         # Save the animation if a path is provided
#         if save_path:
#             ani.save(save_path, writer='pillow', fps=30)
#             print(f"Animation saved to {save_path}")
        
#         # Display the animation
#         plt.show()
        
#         return ani  # Still return the animation object for flexibility

class DubinsCar(Vehicle):
    def __init__(self, theta):
        super().__init__()
        self.state_names = ['x', 'y', 'θ']  # position x, y and heading angle
        self.vehicle_name = "Dubins Car"
        
        # Set parameters
        self.set_dynamics(theta)
        self.reset()
        
        # Initialize history with initial state and zero control
        self.state_history.append(self.s0.copy())
        self.time_history.append(self.current_time)
        self.control_history.append(np.zeros(2))  # Initial zero control [v=0, ω=0]

    def set_dynamics(self, theta):
        # We only need noise parameters and initial state parameters
        self.Gamma = theta.get("Gamma", np.zeros((3, 3)))  # Process noise (optional)
        self.Sigma = theta.get("Sigma", np.eye(3) * 0.01)  # Measurement noise
        self.mu0 = theta.get("mu0", np.zeros(3))  # Initial state mean
        self.V0 = theta.get("V0", np.eye(3) * 0.1)  # Initial state uncertainty

    def reset(self, s0=None):
        super().reset()
        # For Dubins car: 3 states, 3 observable states, 2 controls
        self.Ns = 3  # Number of states
        self.Nx = 3  # Number of observable states
        self.Nu = 2  # Number of controls (v, ω)
        
        self.s0 = self.mu0 + np.random.multivariate_normal(np.zeros(self.Ns), self.V0)
        self.s = self.s0.copy()
        
        # Initialize histories with initial state and zero control
        self.state_history.append(self.s0.copy())
        self.time_history.append(self.current_time)
        # self.control_history.append(np.zeros(2))  # Initial zero control

    def step(self, policy, dt, t):
        # Get the action from the policy (v, ω)
        u = policy.get_action(self.s, t)
        
        # Dubins car dynamics:
        # ẋ = v * cos(θ)
        # ẏ = v * sin(θ)
        # θ̇ = ω
        v, omega = u[0], u[1]
        theta = self.s[2]
        
        sdot = np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            omega
        ])

        # Update the state
        self.s = self.s + sdot * dt + np.random.multivariate_normal(np.zeros(self.Ns), self.Gamma * dt)
        self.x = self.s + np.random.multivariate_normal(np.zeros(self.Nx), self.Sigma * dt)
        
        # Update time
        self.current_time += dt
        
        # Store state, time, and control
        self.state_history.append(self.x.copy())
        self.time_history.append(self.current_time)
        self.control_history.append(u.copy())  

        return self.s, self.x

    def animate(self, save_path=None):
        if len(self.state_history) == 0:
            print("No state history to animate.")
            return
            
        # Convert history lists to numpy arrays
        states = np.array(self.state_history)
        times = np.array(self.time_history)
        
        # Extract position data
        x_positions = states[:, 0]
        y_positions = states[:, 1]
        thetas = states[:, 2]
        
        # Make sure trajectory line has same number of points as states
        trajectory_x = x_positions[:-1]  # Remove last point to match control history
        trajectory_y = y_positions[:-1]  # Remove last point to match control history
        
        # Determine plot bounds
        margin = 1.0
        min_x = np.min(x_positions) - margin
        max_x = np.max(x_positions) + margin
        min_y = np.min(y_positions) - margin
        max_y = np.max(y_positions) + margin
        
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect('equal')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'{self.vehicle_name} Animation')
        ax.grid(True)
        
        # Create triangle patch for the car
        car_length = 0.3
        car = plt.Polygon([[0, 0]], closed=True, fc='blue', ec='black')
        ax.add_patch(car)
        
        # Add trajectory line
        trajectory, = ax.plot([], [], 'r--', alpha=0.5)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        def init():
            car.set_xy(self._get_triangle_coords(x_positions[0], y_positions[0], 
                                               thetas[0], car_length))
            trajectory.set_data([], [])
            time_text.set_text(f'Time: {times[0]:.2f} s')
            return car, trajectory, time_text
        
        def update(frame):
            car.set_xy(self._get_triangle_coords(x_positions[frame], y_positions[frame], 
                                               thetas[frame], car_length))
            # Use only up to current frame for trajectory
            trajectory.set_data(x_positions[:frame], y_positions[:frame])
            time_text.set_text(f'Time: {times[frame]:.2f} s')
            return car, trajectory, time_text
        
        ani = animation.FuncAnimation(
            fig, update, frames=range(len(times)),
            init_func=init, blit=True, interval=50)
        
        if save_path:
            ani.save(save_path, writer='pillow', fps=30)
            print(f"Animation saved to {save_path}")
        
        plt.show()
        return ani

    def _get_triangle_coords(self, x, y, theta, length):
        """Helper function to generate triangle coordinates for car visualization"""
        points = np.array([
            [length, 0],    # nose
            [-length/2, length/2],  # right corner
            [-length/2, -length/2]  # left corner
        ])
        
        # Rotation matrix
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        # Rotate points and translate to position
        rotated = points @ rot.T
        translated = rotated + np.array([x, y])
        
        return translated

    def print_history_info(self):
        """Print information about stored trajectories"""
        print(f"\nHistory Information for {self.vehicle_name}:")
        print(f"Number of time points: {len(self.time_history)}")
        print(f"Number of states: {len(self.state_history)}")
        print(f"Number of controls: {len(self.control_history)}")
        
        if len(self.state_history) > 0:
            print(f"\nState dimension: {len(self.state_history[0])}")
        if len(self.control_history) > 0:
            print(f"Control dimension: {len(self.control_history[0])}")

