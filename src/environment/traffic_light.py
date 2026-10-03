import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from states import LightState, TrafficState
from actions import Actions

class TrafficLightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        max_steps=25,
        max_cars_to_proccess=15,
        max_traffic_count=5,
        stochastic=False,
        seed=42,
        render_mode=None,
    ):
        """
        Initialize the class
        """
        # Maximum steps to take by agent
        self.max_steps = max_steps

        # maximum cars to process
        self.max_cars_to_proccess = max_cars_to_proccess

        # maximum cars present in a single side at any given time
        self.max_traffic_count = max_traffic_count

        # If true, cars arrive in a random way
        self.stochastic = stochastic
        self.seed = seed

        self.render_mode = render_mode

        # Action space
        self.action_space = spaces.Discrete(len(Actions))
        # Observation space
        # Each State is [Light State, North Traffic, South Traffic, East Traffic, West Traffic]
        self.observation_space = spaces.MultiDiscrete([len(LightState)] + [self.max_traffic_count + 1] * 4)

        self.step_count = 0
        self.cars_processed = 0

        # Initial State
        self.light_state = LightState.ALL_RED
        self.traffic_count = np.array([0] * 4) # N, S, E, W

        # Required for render
        self.last_action = None
        self.last_reward = None
        self.action_rejected = False

    def _get_obs(self):
        """Convert internal state to observation format

        Returns:
            np.array: Array with [Light State, N traffic, S traffic, E traffic, W traffic]
        """
        return np.array([
            self.light_state.value, 
            self.traffic_count[0], 
            self.traffic_count[1], 
            self.traffic_count[2], 
            self.traffic_count[3]
        ], dtype=np.int64)

    def _get_info(self):
        """Format internal information about parameters
        
        Returns:
            dict: containing all internal parameters and information
        """
        info = {
                    "max_step_count": self.max_steps,
                    "max_cars_to_proccess": self.max_cars_to_proccess,
                    "max_traffic_count": self.max_traffic_count,
                    "stochastic": self.stochastic,
                    "seed": self.seed,
                    "step_count": self.step_count,
                    "cars_processed": self.cars_processed,
                    "light_state": self.light_state,
                    "traffic_count": self.traffic_count,
                    "action_rejected": self.action_rejected,
                }
        return info

    def reset(self):
        """
        Reset the state of the environment to an initial state
        """
        super().reset(seed=self.seed)

        self.step_count = 0
        self.cars_processed = 0

        self.light_state = LightState.ALL_RED
        self.traffic_count = np.array([0] * 4) # N, S, E, W

        self.last_action = None
        self.last_reward = None
        self.action_rejected = False

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _allowed_light_change(self, next_state):
        allowed_changes = {
            LightState.ALL_RED: {
                LightState.ALL_RED,
                LightState.NS_GREEN,
                LightState.EW_GREEN,
            },
            LightState.NS_GREEN: {
                LightState.NS_GREEN,
                LightState.NS_YELLOW,
            },
            LightState.NS_YELLOW: {
                LightState.ALL_RED,
            },
            LightState.EW_GREEN: {
                LightState.EW_GREEN,
                LightState.EW_YELLOW
            },
            LightState.EW_YELLOW: {
                LightState.ALL_RED,
            }
        }

        return next_state in allowed_changes[self.light_state]

    def _add_car(self):
        """Add a car to the intersection"""
        if self.stochastic:
            pass
        else:
            if self.step_count % 4 == 0:
                self.traffic_count[0] += 1 # North
            elif self.step_count % 4 == 1:
                self.traffic_count[1] += 1 # South
            elif self.step_count % 4 == 2:
                self.traffic_count[2] += 1 # East
            else:
                self.traffic_count[3] += 1 # West

        self.traffic_count = np.minimum(self.traffic_count, self.max_traffic_count)

    def _process_car(self):
        """Remove a car from the intersection if possible
        
        Returns:
            cars_processed: number of cars processed
        """
        cars_processed= 0

        if self.light_state == LightState.NS_GREEN:
            if self.traffic_count[0] > 0:
                self.traffic_count[0] -= 1
                cars_processed += 1

            if self.traffic_count[1] > 0:
                self.traffic_count[1] -= 1
                cars_processed += 1

        elif self.light_state == LightState.EW_GREEN:
            if self.traffic_count[2] > 0:
                self.traffic_count[2] -= 1
                cars_processed += 1

            if self.traffic_count[3] > 0:
                self.traffic_count[3] -= 1
                cars_processed += 1

        self.cars_processed += cars_processed

        return cars_processed

    def _get_reward(self, cars_processed):
        """Calculate the reward based on the number of cars processed and cars waiting.
        +5 for every car passed and -1 for every car in the intersection.

        Args:
            cars_processed: number of cars that passed

        Returns:
            float: reward
        """
        return float(5 * cars_processed - self._get_cars_waiting())

    def _get_traffic_state(self, count):
        """Find the state of the traffic based on traffic count
        
        Args:
            count: traffic count (i.e. number of cars waiting)

        Returns:
            TrafficState: Empty for 0, Light for 1, Moderate for 2, Heavy for 3+ cars
        """
        state = TrafficState.EMPTY

        if count == 1:
            state = TrafficState.LIGHT_TRAFFIC
        elif count == 2:
            state = TrafficState.MODERATE_TRAFFIC
        elif count >= 3:
            state = TrafficState.HEAVY_TRAFFIC
    
        return state

    def _get_traffic_states(self):
        """Convert traffic_count list to traffic states list

        Returns:
            np.Array: Traffic States for each side (North, South, East, West)
        """
        return np.array([self._get_traffic_state(x).value for x in self.traffic_count], dtype=np.int64)

    def _get_cars_waiting(self):
        """Calculate the number of cars waiting at the intersection (all sides).
        
        Returns:
            cars_waiting: the total number of cars waiting at the intersection (all sides)
        """
        cars_waiting = np.sum(self.traffic_count)

        return cars_waiting

    def step(self, action: Actions):
        """
        Execute one timestep within the environment

        Args:
            action: Action to take

        Returns:
            observation: [Light State, N traffic, S traffic, E traffic, W traffic]
            reward: +5 per car processed, -1 per car waiting
            terminated: True once max_cars_to_proccess have been processed
            truncated: True once max_steps steps have been taken
            info: internal parameters
        """
        action = Actions(int(action))
        requested_state = LightState(action.value)

        # Check if the action is allowed. If not, leave light_state unchanged for the step
        is_allowed = self._allowed_light_change(requested_state)
        if is_allowed:
            self.light_state = requested_state

        cars_processed = self._process_car()

        # reqard will be +5 for each car processed and -1 for each car waiting
        reward = self._get_reward(cars_processed)

        # Add more cars to environment
        self._add_car()
        self.step_count += 1

        terminated = self.cars_processed >= self.max_cars_to_proccess
        truncated = self.step_count >= self.max_steps

        self.last_action = action
        self.last_reward = reward
        self.action_rejected = not is_allowed

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _signal_color(self, direction):
        """Color of the light shown for `direction`

        Args:
            `direction`: Can be "N", "S", "E", "W"

        Returns:
            color: "red", "yellow" or "green"
        """
        ns = direction == "N" or direction == "S"

        if self.light_state == LightState.NS_GREEN:
            return "green" if ns else "red"
        if self.light_state == LightState.NS_YELLOW:
            return "yellow" if ns else "red"
        if self.light_state == LightState.EW_GREEN:
            return "red" if ns else "green"
        if self.light_state == LightState.EW_YELLOW:
            return "red" if ns else "yellow"

        return "red"

    def render(self):
        """
        Visualize the environment

        Draws the 4x4 intersection next to a panel with the step, action,
        reward and whether that action was rejected as an illegal change.

        Returns:
            np.array: RGB frame if render_mode is "rgb_array", else None
        """
        # Positions of entering sides and lights for each side in the 4x4 grid
        sides = {(0, 1): "N", (2, 0): "W", (1, 3): "E", (3, 2): "S"}
        lights = {(1, 1): "N", (2, 1): "W", (1, 2): "E", (2, 2): "S"}

        # Indices for traffic_count
        index = {"N": 0, "S": 1, "E": 2, "W": 3}

        #                   EMPTY       LIGHT       MODERATE        HEAVY
        traffic_shades = ["#ffffff", "#94b7d2", "#4b80ae", "#073e75"]
        lamps = {"red": "#ff0000", "yellow": "#ffbf00", "green": "#00ff2f"}

        fig, (ax, panel) = plt.subplots(
            1, 2, figsize=(8, 4.2), gridspec_kw={"width_ratios": [3, 1.7]}
        )
        ax.set_xlim(0, 4)
        ax.set_ylim(4, 0) # row 0 at the top
        ax.set_aspect("equal")
        ax.axis("off")

        for row in range(4):
            for col in range(4):
                cell = (row, col)

                if cell in sides:
                    count = int(self.traffic_count[index[sides[cell]]])
                    state = self._get_traffic_state(count)
                    color = traffic_shades[state.value]
                elif row in (0, 3) and col in (0, 3):
                    color = "#7fb069" # grass (cornrs)
                else:
                    color = "#4a4a4a" # road

                ax.add_patch(plt.Rectangle((col, row), 1, 1, facecolor=color,
                                           edgecolor="white", linewidth=1.5))

                if cell in sides:
                    label = f"{sides[cell]}: {count}\n{state.name.replace('_TRAFFIC', '')}"
                    ax.text(col + 0.5, row + 0.5, label, ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if state.value >= 2 else "black")
                elif cell in lights:
                    ax.add_patch(plt.Circle(
                        (col + 0.5, row + 0.5), 0.22, edgecolor="black",
                        facecolor=lamps[self._signal_color(lights[cell])]))
                    ax.text(col + 0.5, row + 0.85, lights[cell], ha="center",
                            va="center", fontsize=8, color="#dddddd")

        if self.cars_processed >= self.max_cars_to_proccess:
            status = "TERMINATED (quota met)"
        elif self.step_count >= self.max_steps:
            status = "TRUNCATED (time limit)"
        else:
            status = "RUNNING"

        action = self.last_action.name if self.last_action is not None else "--"
        reward = f"{self.last_reward:+.1f}" if self.last_reward is not None else "--"

        panel.axis("off")
        panel.text(0, 0.95, "\n".join([
            f"step       {self.step_count} / {self.max_steps}",
            f"action     {action}",
            f"reward     {reward}",
            f"light      {self.light_state.name}",
            f"processed  {self.cars_processed} / {self.max_cars_to_proccess}",
            f"waiting    {int(self._get_cars_waiting())} cars",
            f"status     {status}",
        ]), va="top", family="monospace", fontsize=9, linespacing=1.8)

        # Show that an illegal action was blocked
        if self.action_rejected:
            panel.text(0, 0.13, "ILLEGAL TRANSITION\nlight unchanged", va="top",
                       family="monospace", fontsize=9, fontweight="bold",
                       color="#e03131")

        fig.tight_layout()

        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            plt.close(fig)
            return frame

        plt.show()
        plt.close(fig)
        return None
