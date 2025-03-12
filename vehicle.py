# Vehicle Definitions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Vehicle:
    def __init__(self):
        self.state_history = []
        self.time_history = []
        self.current_time = 0
        self.state_names = []

    def reset(self):
        self.state_history = []
        self.time_history = []
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
        if len(self.state_history) == 0:
            print("No state history to plot.")
            return
            
        # Convert history lists to numpy arrays for easier manipulation
        states = np.array(self.state_history)
        times = np.array(self.time_history)
        
        # Get number of state variables
        n_states = states.shape[1]
        
        # Create subplots, one for each state variable
        fig, axes = plt.subplots(n_states, 1, figsize=figsize, sharex=True)
        if n_states == 1:
            axes = [axes]  # Make sure axes is iterable even with one subplot
            
        # Plot each state variable
        for i in range(n_states):
            axes[i].plot(times, states[:, i])
            axes[i].set_ylabel(self.state_names[i])
            axes[i].grid(True)
            
        # Add labels and title
        axes[-1].set_xlabel('Time')
        if title:
            fig.suptitle(title)
        
        plt.tight_layout()
        plt.show()

class MassSpringDamper(Vehicle):
    def __init__(self, s0, p):
        super().__init__()
        self.s0 = s0
        self.s = self.s0
        self.state_names = ['x1', 'x2', 'Fk1', 'Fk2']
        
        # Constants for the system (these should be defined before using them in A and B)
        self.M1 = p[0]  # mass 1
        self.M2 = p[1]  # mass 2
        self.K1 = p[2]  # spring constant 1
        self.K2 = p[3]  # spring constant 2
        self.B = p[4]  # damping constant

        # Underlying Dynamics
        self.A = np.array([[-1/self.M1, 0, -self.B/self.M1, self.B/self.M1],
                       [0, 0, 0, -1/self.M2],
                       [self.K1, 0, 0, 0],
                       [-self.K2, self.K2, 0, 0]])
        
        self.B = np.array([[0, 1/self.M1],
                      [1/self.M2, 0],
                      [0, -self.K1],
                      [0, 0]])
        
        self.C = np.eye(4)
        self.D = np.zeros((4, 2))
        
        # Initialize history with initial state
        self.state_history.append(self.s0.copy())
        self.time_history.append(self.current_time)

    def reset(self):
        super().reset()
        self.s = self.s0.copy()
        self.state_history.append(self.s.copy())
        self.time_history.append(self.current_time)

    def step(self, policy, dt):
        # Get the action from the policy
        u = policy.get_action(self.s)

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
    
    def animate(self):
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
        
        # Determine the range for plotting
        min_pos = min(np.min(x1_positions), np.min(x2_positions)) - 1
        max_pos = max(np.max(x1_positions), np.max(x2_positions)) + 1
        
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_xlim(min_pos, max_pos)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Position')
        ax.set_title('Mass-Spring-Damper System Animation')
        ax.grid(True)
        
        # Create objects for animation
        mass1 = plt.Rectangle((x1_positions[0]-0.2, -0.5), 0.4, 0.4, fc='blue', ec='black')
        mass2 = plt.Rectangle((x2_positions[0]-0.2, -0.5), 0.4, 0.4, fc='red', ec='black')
        spring = plt.Line2D([0, 0], [0, 0], color='green', linewidth=2, linestyle='-')
        left_spring = plt.Line2D([0, 0], [0, 0], color='green', linewidth=2, linestyle='-')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        # Add elements to the plot
        ax.add_patch(mass1)
        ax.add_patch(mass2)
        ax.add_line(spring)
        ax.add_line(left_spring)
        
        # Function to initialize the animation
        def init():
            mass1.set_xy((x1_positions[0]-0.2, -0.5))
            mass2.set_xy((x2_positions[0]-0.2, -0.5))
            spring.set_data([x1_positions[0], x2_positions[0]], [-0.3, -0.3])
            left_spring.set_data([min_pos, x1_positions[0]], [-0.3, -0.3])
            time_text.set_text(f'Time: {times[0]:.2f} s')
            return mass1, mass2, spring, left_spring, time_text
        
        # Function to update the animation for each frame
        def update(frame):
            mass1.set_xy((x1_positions[frame]-0.2, -0.5))
            mass2.set_xy((x2_positions[frame]-0.2, -0.5))
            spring.set_data([x1_positions[frame], x2_positions[frame]], [-0.3, -0.3])
            left_spring.set_data([min_pos, x1_positions[frame]], [-0.3, -0.3])
            time_text.set_text(f'Time: {times[frame]:.2f} s')
            return mass1, mass2, spring, left_spring, time_text
        
        # Create the animation
        ani = animation.FuncAnimation(
            fig, update, frames=range(len(times)),
            init_func=init, blit=True, interval=50)
        
        plt.close()  # Prevent duplicate display in Jupyter notebooks
        
        return ani  # Return the animation object so it can be displayed or saved
    
    
