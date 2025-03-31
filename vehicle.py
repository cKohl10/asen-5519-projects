# Vehicle Definitions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Vehicle:
    def __init__(self):
        self.state_history = []
        self.time_history = []
        self.control_history = []
        self.current_time = 0
        self.state_names = []

    def reset(self):
        self.state_history = []
        self.time_history = []
        self.control_history = []
        self.current_time = 0

    def step(self, policy):
        pass

    def animate(self):
        pass
    
    def plot_states(self, title=None, figsize=(12, 8)):
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
        controls = np.array(self.control_history)
        times = np.array(self.time_history)
        
        # Get number of state and control variables
        n_states = states.shape[1]
        n_controls = controls.shape[1]

        # Create figure for states
        fig_states, axes_states = plt.subplots(n_states, 1, figsize=figsize, sharex=True)
        if n_states == 1:
            axes_states = [axes_states]
            
        # Plot each state variable
        for i in range(n_states):
            axes_states[i].plot(times, states[:, i])
            axes_states[i].set_ylabel(self.state_names[i])
            axes_states[i].grid(True)
            
        # Add labels and title for states
        axes_states[-1].set_xlabel('Time')
        fig_states.suptitle("States" if not title else f"States - {title}")
        
        # Create separate figure for controls
        fig_controls, axes_controls = plt.subplots(n_controls, 1, figsize=figsize, sharex=True)
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

class MassSpringDamper(Vehicle):
    def __init__(self, s0, p):
        super().__init__()
        self.s0 = s0
        self.s = self.s0
        self.state_names = ['x1', 'x2', 'v1', 'v2', 'Fk1', 'Fk2'] # x1 and x2 are the positions of the masses, v1 and v2 are the velocities of the masses, Fk1 and Fk2 are the forces exerted by the springs
        
        # Constants for the system (these should be defined before using them in A and B)
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

        # Underlying Dynamics
        
        self.C = np.eye(6)
        self.D = np.zeros((6, 2))
        
        # Initialize history with initial state
        self.state_history.append(self.s0.copy())
        self.time_history.append(self.current_time)

    def reset(self):
        super().reset()
        self.s = self.s0.copy()
        self.state_history.append(self.s.copy())
        self.time_history.append(self.current_time)

    def step(self, policy, dt, t):

        # Check for collisions with velocity direction consideration
        x1, x2 = self.s[0], self.s[1]
        v1, v2 = self.s[2], self.s[3]
        
        if np.abs(x1 - x2) < 2*self.r and (v2 - v1) < 0:  # Only collide when approaching
            # Calculate new velocities using physics of collisions
            m1, m2 = self.M1, self.M2
            e = self.COR
            
            # Conservation of momentum with COR
            new_v1 = ((m1 - e*m2)*v1 + (1 + e)*m2*v2) / (m1 + m2)
            new_v2 = ((1 + e)*m1*v1 + (m2 - e*m1)*v2) / (m1 + m2)
            
            self.s[2] = new_v1
            self.s[3] = new_v2

        if x1 <= self.r and v1 <= 0:
            self.s[2] = -self.s[2]
        if x2 <= self.r and v2 <= 0:
            self.s[3] = -self.s[3]

        # Get the action from the policy
        u = policy.get_action(self.s, t)
        self.control_history.append(u.copy())

        # Step the dynamics
        sdot = self.A @ self.s + self.B @ u

        # Update the state
        self.s = self.s + sdot * dt
        
        # Update time
        self.current_time += dt
        
        # Store state and time in history
        self.state_history.append(self.s.copy())
        self.time_history.append(self.current_time)

        # Return the next state
        return self.s
    
    def animate(self, save_path=None):
        # This problem is two masses connected by springs and dampers.
        # x1 and x2 are the positions of the masses.
        if len(self.state_history) == 0:
            print("No state history to animate.")
            return
            
        # Convert history lists to numpy arrays
        states = np.array(self.state_history)
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
        ax.set_title('Mass-Spring-Damper System Animation')
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
    
    
