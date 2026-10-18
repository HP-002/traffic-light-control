import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from states import LightState, TrafficState, ExtraTimeState
from actions import Actions

class TrafficLightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Number of sides of the intersection (N, S, E, W)
    NUM_SIDES = 4

    # Maps LightState to (index of the light in traffic_count, True if light is Green).
    traffic_light_index = {
        LightState.N_GREEN: (0, True),
        LightState.N_YELLOW: (0, False), 
        LightState.S_GREEN: (1, True),
        LightState.S_YELLOW: (1, False),
        LightState.E_GREEN: (2, True),
        LightState.E_YELLOW: (2, False),
        LightState.W_GREEN: (3, True),
        LightState.W_YELLOW: (3, False),
    }

    def __init__(
        self,
        max_steps=25,
        max_cars_to_process=15,
        max_traffic_count=5,
        arrival_rate=0.3,
        deterministic_arrival_count=2,
        car_process_count=(2, 1),
        render_fps=4,
        illegal_action_penalty=-10,
        car_process_reward=5,
        car_waiting_reward=-1,
        traffic_thresholds=(1, 2),
        time_threshold=3,
        penalty_factor=0.95,
        stochastic=False,
        seed=42,
        render_mode=None,
    ):
        """Initialize the environment

        Args:
            max_steps: steps before the episode is truncated
            max_cars_to_process: cars to process before the episode is terminated
            max_traffic_count: maximum cars waiting on a single side
            arrival_rate: Poisson rate of arriving cars when stochastic. A number for all sides or a list of 4 values (N, S, E, W)
            deterministic_arrival_count: cars added to one side each step when not stochastic
            car_process_count: max cars processed in one step as (green, yellow)
            render_fps: frames per second for rendering
            illegal_action_penalty: reward for an illegal light change
            car_process_reward: reward per car processed
            car_waiting_reward: reward per car waiting on a red side
            traffic_thresholds: max cars for (LIGHT, MODERATE) traffic. HEAVY above that.
                Must satisfy 1 <= LIGHT < MODERATE < max_traffic_count so every state is reachable
            time_threshold: consecutive steps one side can be green or yellow (counted together)
                before the processing reward starts to decay and ExtraTimeState flips to ABOVE
            penalty_factor: fraction of the processing reward kept per step above time_threshold
                (0.95 keeps 95%, i.e. 5% decay per step)
            stochastic: if True, cars arrive randomly, else in a fixed rotation
            seed: seed for the random number generator
            render_mode: "human", "rgb_array" or None

        Raises:
            ValueError: if arrival_rate is a list without 4 values, car_process_count is not a
                (green, yellow) pair, or traffic_thresholds is not a valid pair
        """
        # Maximum steps to take by agent
        self.max_steps = max_steps

        # maximum cars to process
        self.max_cars_to_process = max_cars_to_process

        # Arrival rate for cars on each side (N, S, E, W)
        if np.isscalar(arrival_rate):
            self.arrival_rate = [arrival_rate] * self.NUM_SIDES
        else:
            if len(arrival_rate) != self.NUM_SIDES:
                raise ValueError(f"arrival_rate must be a number or a list of {self.NUM_SIDES} values, got {arrival_rate}")
            self.arrival_rate = list(arrival_rate)

        # Number of cars added to one side each step when not stochastic
        self.deterministic_arrival_count = deterministic_arrival_count

        # maximum cars present in a single side at any given time
        self.max_traffic_count = max_traffic_count

        # Number of cars to process when light is green and yellow
        if np.isscalar(car_process_count) or len(car_process_count) != 2:
            raise ValueError(f"car_process_count must be a (green, yellow) pair, got {car_process_count}")
        self.car_process_count_green = car_process_count[0]
        self.car_process_count_yellow = car_process_count[1]

        # Used to bucketize the traffic into LIGHT_TRAFFIC, MODERATE_TRAFFIC, and HEAVY_TRAFFIC
        if np.isscalar(traffic_thresholds) or len(traffic_thresholds) != 2:
            raise ValueError(f"traffic_thresholds must be a (LIGHT, MODERATE) pair, got {traffic_thresholds}")
        if not 1 <= traffic_thresholds[0] < traffic_thresholds[1] < max_traffic_count:
            raise ValueError(
                f"traffic_thresholds must satisfy 1 <= LIGHT < MODERATE < max_traffic_count ({max_traffic_count}), "
                f"got {traffic_thresholds}"
            )
        self.traffic_thresholds = tuple(traffic_thresholds)

        # If true, cars arrive in a random way
        self.stochastic = stochastic
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

        self.render_mode = render_mode
        # Copy so the class-level metadata shared by all instances is not modified
        self.metadata = {**self.metadata, "render_fps": render_fps}

        # Action space
        self.action_space = spaces.Discrete(len(Actions))
        # Observation space
        # Each State is [Light State, North Traffic, South Traffic, East Traffic, West Traffic, ExtraTime in Green/Yellow]
        self.observation_space = spaces.MultiDiscrete([len(LightState)] + [len(TrafficState)] * self.NUM_SIDES + [len(ExtraTimeState)])

        self.step_count = 0
        self.cars_processed = 0

        # Initial State
        self.light_state = LightState.ALL_RED
        self.traffic_count = np.array([0] * self.NUM_SIDES) # N, S, E, W

        # Used for calculating rewards
        self.illegal_action_penalty = illegal_action_penalty
        self.car_process_reward = car_process_reward
        self.car_waiting_reward = car_waiting_reward
        self.time_threshold = time_threshold
        self.penalty_factor = penalty_factor # Penalty factor for reducing reward time spent in green/yellow > time_threshold
        self.current_time_in_gy = 0

        # Required for render
        self.last_action = None
        self.last_reward = None
        self.action_rejected = False

    def _get_obs(self):
        """Convert internal state to observation format

        Returns:
            np.array: Array with [Light State, N traffic, S traffic, E traffic, W traffic, Extra time]
        """
        traffic_states = self._get_traffic_states()
        extra_time_state = self._get_extra_time_state()
        return np.array([
            self.light_state.value, 
            traffic_states[0], 
            traffic_states[1], 
            traffic_states[2], 
            traffic_states[3],
            extra_time_state,
        ], dtype=np.int64)

    def _get_info(self):
        """Format internal information about parameters
        
        Returns:
            dict: containing all internal parameters and information
        """
        info = {
                    "max_step_count": self.max_steps,
                    "max_cars_to_process": self.max_cars_to_process,
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

    def reset(self, *, seed=None, options=None):
        """
        Reset the state of the environment to an initial state
        """
        super().reset(seed=seed)

        # Re-seed the arrival RNG only when a seed is passed, otherwise keep the existing stream
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed=seed)

        self.step_count = 0
        self.cars_processed = 0

        self.light_state = LightState.ALL_RED
        self.traffic_count = np.array([0] * self.NUM_SIDES) # N, S, E, W

        self.current_time_in_gy = 0

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
                LightState.N_GREEN, LightState.S_GREEN,
                LightState.E_GREEN, LightState.W_GREEN,
            },
            LightState.N_GREEN: {
                LightState.N_GREEN,
                LightState.N_YELLOW,
            },
            LightState.N_YELLOW: {
                LightState.ALL_RED,
            },
            LightState.S_GREEN: {
                LightState.S_GREEN,
                LightState.S_YELLOW,
            },
            LightState.S_YELLOW: {
                LightState.ALL_RED,
            },
            LightState.E_GREEN: {
                LightState.E_GREEN,
                LightState.E_YELLOW,
            },
            LightState.E_YELLOW: {
                LightState.ALL_RED,
            },
            LightState.W_GREEN: {
                LightState.W_GREEN,
                LightState.W_YELLOW,
            },
            LightState.W_YELLOW: {
                LightState.ALL_RED,
            },
        }

        return next_state in allowed_changes[self.light_state]

    def _add_car(self):
        """Add a car to the intersection
        - If stochastic, cars are added to each side based on a Poisson distribution with that side's rate in `self.arrival_rate`.
        - If deterministic, `self.deterministic_arrival_count` cars are added to one side of the intersection each round on a rotational basis (N, S, E, W).
        """
        if self.stochastic:
            self.traffic_count += self.rng.poisson(self.arrival_rate)
        else:
            self.traffic_count[self.step_count % self.NUM_SIDES] += self.deterministic_arrival_count

        self.traffic_count = np.minimum(self.traffic_count, self.max_traffic_count)

    def _process_car(self):
        """Remove cars from the intersection if possible. For any side:
        - If the side is GREEN, at most `car_process_count_green` cars are processed.
        - If the side is YELLOW, at most `car_process_count_yellow` cars are processed.
        
        Returns:
            cars_to_process: number of cars processed or removed
        """
        # No car to process
        cars_to_process = 0

        idx, is_green = self.traffic_light_index.get(self.light_state, (None, False))

        # if not all RED, process cars on the NON-RED side
        if idx is not None:
            cars_to_process = self.car_process_count_yellow
            if is_green:
                cars_to_process = self.car_process_count_green

            cars_to_process = min(cars_to_process, self.traffic_count[idx])
            self.traffic_count[idx] -= cars_to_process

        self.cars_processed += cars_to_process
        return cars_to_process

    def _update_time_in_green(self, prev_light):
        """Check consecutive time the light is green/yellow for same side. Otherwise resets to 0."""
        idx, _ = self.traffic_light_index.get(self.light_state, (None, False))
        prev_idx, _ = self.traffic_light_index.get(prev_light, (None, False))
        if idx is None:
            self.current_time_in_gy = 0
        elif idx == prev_idx:
            self.current_time_in_gy += 1
        else:
            self.current_time_in_gy = 1

    def _get_reward(self, cars_processed, cars_waiting, is_legal):
        """Calculate the reward based on the number of cars processed, number of cars waiting, and action.

        Args:
            cars_processed: number of cars that passed
            cars_waiting: number of cars waiting on the red sides
            is_legal: True if the action was legal, False otherwise

        Returns:
            float: immediate reward
        """
        # Decay starts from the threshold so the reward does not jump when it is crossed
        time = max(0, self.current_time_in_gy - self.time_threshold)
        r = 0

        if not is_legal:
            r = self.illegal_action_penalty

        r = (
                r +
                (self.penalty_factor ** time) * self.car_process_reward * cars_processed +
                self.car_waiting_reward * cars_waiting
            )

        return float(r)

    def _get_traffic_state(self, count):
        """Find the state of the traffic based on traffic count
        
        Args:
            count: traffic count (i.e. number of cars waiting)

        Returns:
            TrafficState: EMPTY for 0, LIGHT up to traffic_thresholds[0], MODERATE up to
                traffic_thresholds[1], HEAVY above that
        """
        state = TrafficState.EMPTY

        if count == 0:
            state = TrafficState.EMPTY
        elif count <= self.traffic_thresholds[0]:
            state = TrafficState.LIGHT_TRAFFIC
        elif count <= self.traffic_thresholds[1]:
            state = TrafficState.MODERATE_TRAFFIC
        else:
            state = TrafficState.HEAVY_TRAFFIC
    
        return state

    def _get_traffic_states(self):
        """Convert traffic_count list to traffic states list

        Returns:
            np.Array: Traffic States for each side (North, South, East, West)
        """
        return np.array([self._get_traffic_state(x).value for x in self.traffic_count], dtype=np.int64)

    def _get_extra_time_state(self):
        """Return the ExtraTimeState based on the current time in green or yellow (`current_time_in_gy`).
        
        Returns:
            ExtraTimeState: BELOW if current_time_in_gy <= time_threshold and ABOVE otherwise
        """
        return ExtraTimeState.BELOW.value if self.current_time_in_gy <= self.time_threshold else ExtraTimeState.ABOVE.value

    def _get_cars_waiting(self):
        """Calculate the number of cars waiting at the intersection (red sides only).
        
        Returns:
            cars_waiting: the total number of cars waiting at the intersection (red sides only)
        """
        cars_waiting = int(np.sum(self.traffic_count))

        idx, _ = self.traffic_light_index.get(self.light_state, (None, False))
        if idx is not None:
            cars_waiting -= int(self.traffic_count[idx])

        return cars_waiting

    def step(self, action: Actions):
        """
        Execute one timestep within the environment

        Args:
            action: Action to take

        Returns:
            observation: [Light State, N traffic, S traffic, E traffic, W traffic, Extra time]
            reward: immediate reward
            terminated: True once max_cars_to_process have been processed
            truncated: True once max_steps steps have been taken
            info: internal parameters
        """
        action = Actions(int(action))
        requested_state = LightState(action.value)

        prev_light = self.light_state

        # Check if the action is allowed. If not, leave light_state unchanged for the step
        is_allowed = self._allowed_light_change(requested_state)
        if is_allowed:
            self.light_state = requested_state

        cars_processed = self._process_car()
        cars_waiting = self._get_cars_waiting()

        self._update_time_in_green(prev_light=prev_light)

        reward = self._get_reward(cars_processed, cars_waiting, is_allowed)

        # Add more cars to environment
        self._add_car()
        self.step_count += 1

        terminated = self.cars_processed >= self.max_cars_to_process
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
        current_light = self.light_state

        if direction == "N":
            if current_light == LightState.N_GREEN:
                return "green"
            elif current_light == LightState.N_YELLOW:
                return "yellow"
        elif direction == "S":
            if current_light == LightState.S_GREEN:
                return "green"
            elif current_light == LightState.S_YELLOW:
                return "yellow"
        elif direction == "E":
            if current_light == LightState.E_GREEN:
                return "green"
            elif current_light == LightState.E_YELLOW:
                return "yellow"
        elif direction == "W":
            if current_light == LightState.W_GREEN:
                return "green"
            elif current_light == LightState.W_YELLOW:
                return "yellow"

        return "red"

    def render(self):
        """
        Visualize the environment

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
                            color="white" if state.value >= TrafficState.MODERATE_TRAFFIC.value else "black")
                elif cell in lights:
                    ax.add_patch(plt.Circle(
                        (col + 0.5, row + 0.5), 0.22, edgecolor="black",
                        facecolor=lamps[self._signal_color(lights[cell])]))
                    ax.text(col + 0.5, row + 0.85, lights[cell], ha="center",
                            va="center", fontsize=8, color="#dddddd")

        if self.cars_processed >= self.max_cars_to_process:
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
            f"processed  {self.cars_processed} / {self.max_cars_to_process}",
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
