import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# Colors for each theme. The car colors are an ordinal ramp (one hue) checked against the road color:
# in "light" darker cars waited longer, in "dark" brighter cars waited longer.
THEMES = {
    "light": {
        "grass": "#39a602", "road": "#4e4f50", "lane_mark": "#ffffff",
        "sidewalk": "#e6d5b0", "curb": "#c2ab84",
        "card": "#ffffff", "track": "#e9ecef",
        "ink": "#212529", "ink_secondary": "#495057", "ink_muted": "#868e96",
        "accent": "#0c8599", "zero_line": "#dee2e6", "signal_head": "#343a40",
        "cars": ["#edf2ff", "#bac8ff", "#748ffc", "#3b5bdb", "#1c2f8c"],
    },
    "dark": {
        "grass": "#1f3b29", "road": "#2d3035", "lane_mark": "#868e96",
        "sidewalk": "#4a4335", "curb": "#5c5240",
        "card": "#262b31", "track": "#343a40",
        "ink": "#f1f3f5", "ink_secondary": "#ced4da", "ink_muted": "#868e96",
        "accent": "#3bc9db", "zero_line": "#495057", "signal_head": "#101316",
        "cars": ["#5c6fb8", "#7f95f0", "#a8b9ff", "#d6deff", "#f0f3ff"],
    },
}
CARD_ALPHA = 0.96

# Signal colors are the same in both themes and only used for signals (and the illegal action badge)
LAMP_ON = {"red": "#ff0004", "yellow": "#ffb703", "green": "#00ff55"}
LAMP_OFF = {"red": "#ff0004", "yellow": "#ffb703", "green": "#00ff55"}
CRITICAL = "#e03131"

# Lower edge (in steps waited) of each car color bin, one bin per color in THEMES[...]["cars"]
WAIT_BINS = [0, 10, 25, 50, 100]
# Number of most recent steps shown in the reward line
REWARD_WINDOW = 50

# Frame size. Fixed DPI so every frame has the same pixel size (600 x 600)
FIGSIZE = (6, 6)
DPI = 100

# Layout in world units. The frame is WORLD_SIZE x WORLD_SIZE with y pointing up.
WORLD_SIZE = 10.0
ROAD_START, ROAD_END = 4.0, 6.0         # edges of both roads
ROAD_MID = (ROAD_START + ROAD_END) / 2
LANE_WIDTH = (ROAD_END - ROAD_START) / 2
SIDEWALK_WIDTH = 0.3
CAR_LENGTH, CAR_GAP = 0.34, 0.08
MAX_CARS_DRAWN = 8                      # cars drawn per lane, the rest are shown as "+N"
CARD_WIDTH, CARD_HEIGHT, CARD_MARGIN, CARD_PADDING = 3.5, 1.75, 0.15, 0.18
BADGE_WIDTH, BADGE_HEIGHT = 1.3, 0.78
LAMP_SPACING, LAMP_RADIUS = 0.22, 0.08

# Index of each side in traffic_count and queues
SIDE_INDEX = {"N": 0, "S": 1, "E": 2, "W": 3}

# Approach lane of each side (cars drive on the right):
#   stop: end points of the stop line
#   start: center of the first car slot
#   direction: direction the queue grows in, away from the intersection
APPROACHES = {
    "N": {"stop": ((ROAD_START, ROAD_END), (ROAD_MID, ROAD_END)),
          "start": (ROAD_START + LANE_WIDTH / 2, ROAD_END + 0.12), "direction": (0, 1)},
    "S": {"stop": ((ROAD_MID, ROAD_START), (ROAD_END, ROAD_START)),
          "start": (ROAD_MID + LANE_WIDTH / 2, ROAD_START - 0.12), "direction": (0, -1)},
    "E": {"stop": ((ROAD_END, ROAD_MID), (ROAD_END, ROAD_END)),
          "start": (ROAD_END + 0.12, ROAD_MID + LANE_WIDTH / 2), "direction": (1, 0)},
    "W": {"stop": ((ROAD_START, ROAD_START), (ROAD_START, ROAD_MID)),
          "start": (ROAD_START - 0.12, ROAD_START + LANE_WIDTH / 2), "direction": (-1, 0)},
}

# Signal head (center, horizontal) and count badge center for each side, on the corner before the stop line
SIGNALS = {
    "N": ((ROAD_START - 0.3, ROAD_END + 0.5), False, (ROAD_START - 1.25, ROAD_END + 0.55)),
    "S": ((ROAD_END + 0.3, ROAD_START - 0.5), False, (ROAD_END + 1.25, ROAD_START - 0.55)),
    "E": ((ROAD_END + 0.5, ROAD_END + 0.3), True, (ROAD_END + 0.55, ROAD_END + 1.2)),
    "W": ((ROAD_START - 0.5, ROAD_START - 0.3), True, (ROAD_START - 0.55, ROAD_START - 1.2)),
}


def _rounded_box(ax, x, y, width, height, color, radius=0.15, zorder=5, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder))


def _progress_bar(ax, theme, x, y, width, fraction, height=0.13):
    _rounded_box(ax, x, y, width, height, theme["track"], radius=height / 2, zorder=6)
    if fraction > 0:
        filled = max(width * min(fraction, 1), height)
        _rounded_box(ax, x, y, filled, height, theme["accent"], radius=height / 2, zorder=7)


def _key_value(ax, theme, x, y, width, key, value, fontsize=9):
    """Muted key on the left and bold value on the right of a card row"""
    ax.text(x, y, key, fontsize=6.5, color=theme["ink_muted"], va="center", zorder=8)
    ax.text(x + width, y, value, fontsize=fontsize, color=theme["ink"], fontweight="bold",
            ha="right", va="center", zorder=8)


def _card(ax, theme, x, y):
    _rounded_box(ax, x, y, CARD_WIDTH, CARD_HEIGHT, theme["card"], zorder=5, alpha=CARD_ALPHA)


def _car_color(theme, waiting_time):
    return theme["cars"][np.searchsorted(WAIT_BINS, waiting_time, side="right") - 1]


def _signal_head(ax, theme, x, y, color, horizontal):
    """3-lamp signal head centered at (x, y) with only `color` lit"""
    long_side, short_side = 3 * LAMP_SPACING + 0.1, LAMP_SPACING + 0.1
    width, height = (long_side, short_side) if horizontal else (short_side, long_side)
    _rounded_box(ax, x - width / 2, y - height / 2, width, height, theme["signal_head"], radius=0.07, zorder=4)

    for i, lamp in enumerate(["red", "yellow", "green"]):
        offset = (i - 1) * LAMP_SPACING
        lamp_x, lamp_y = (x + offset, y) if horizontal else (x, y - offset)
        ax.add_patch(plt.Circle((lamp_x, lamp_y), LAMP_RADIUS, zorder=5,
                                facecolor=LAMP_ON[lamp] if lamp == color else LAMP_OFF[lamp]))


def _draw_roads(ax, theme):
    # Sidewalks are drawn wider than the roads, then the roads are drawn over their middle
    road_width = ROAD_END - ROAD_START
    for x, y, w, h, color in [
        (ROAD_START - SIDEWALK_WIDTH, 0, road_width + 2 * SIDEWALK_WIDTH, WORLD_SIZE, theme["sidewalk"]),
        (0, ROAD_START - SIDEWALK_WIDTH, WORLD_SIZE, road_width + 2 * SIDEWALK_WIDTH, theme["sidewalk"]),
        (ROAD_START, 0, road_width, WORLD_SIZE, theme["road"]),
        (0, ROAD_START, WORLD_SIZE, road_width, theme["road"]),
    ]:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="none"))

    # Curbs along the road edges, outside the intersection box
    for edge in (ROAD_START, ROAD_END):
        for xs, ys in [([edge, edge], [0, ROAD_START]), ([edge, edge], [ROAD_END, WORLD_SIZE]),
                       ([0, ROAD_START], [edge, edge]), ([ROAD_END, WORLD_SIZE], [edge, edge])]:
            ax.plot(xs, ys, color=theme["curb"], lw=2)

    # Lane dividers
    dash = {"color": theme["lane_mark"], "lw": 1, "ls": (0, (4, 4)), "alpha": 0.8}
    ax.plot([ROAD_MID, ROAD_MID], [0, ROAD_START], **dash)
    ax.plot([ROAD_MID, ROAD_MID], [ROAD_END, WORLD_SIZE], **dash)
    ax.plot([0, ROAD_START], [ROAD_MID, ROAD_MID], **dash)
    ax.plot([ROAD_END, WORLD_SIZE], [ROAD_MID, ROAD_MID], **dash)


def _draw_approaches(ax, theme, env):
    """Stop lines in the signal color, queued cars colored by waiting time, signal heads and count badges"""
    for side, approach in APPROACHES.items():
        signal = env._signal_color(side)
        (x0, y0), (x1, y1) = approach["stop"]
        ax.plot([x0, x1], [y0, y1], color=LAMP_ON[signal], lw=4, solid_capstyle="butt", zorder=2)

        start_x, start_y = approach["start"]
        dx, dy = approach["direction"]
        car_width = LANE_WIDTH * 0.6
        width, height = (CAR_LENGTH, car_width) if dx else (car_width, CAR_LENGTH)

        queue = env.queues[SIDE_INDEX[side]]
        for i, car in enumerate(list(queue)[:MAX_CARS_DRAWN]):
            offset = CAR_LENGTH / 2 + i * (CAR_LENGTH + CAR_GAP)
            x, y = start_x + dx * offset, start_y + dy * offset
            ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                        boxstyle="round,pad=0,rounding_size=0.07",
                                        facecolor=_car_color(theme, car.waiting_time),
                                        edgecolor=theme["road"], lw=1, zorder=3))

        if len(queue) > MAX_CARS_DRAWN:
            offset = MAX_CARS_DRAWN * (CAR_LENGTH + CAR_GAP) + 0.28
            ax.text(start_x + dx * offset, start_y + dy * offset, f"+{len(queue) - MAX_CARS_DRAWN}",
                    ha="center", va="center", fontsize=7.5, fontweight="bold", color=theme["ink"], zorder=8,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": theme["card"], "edgecolor": "none"})

        (head_x, head_y), horizontal, (badge_x, badge_y) = SIGNALS[side]
        _signal_head(ax, theme, head_x, head_y, signal, horizontal)

        # Badge: count out of the cap, a fill bar and the TrafficState the agent observes
        count = int(env.traffic_count[SIDE_INDEX[side]])
        state = env._get_traffic_state(count).name.replace("_TRAFFIC", "")
        left = badge_x - BADGE_WIDTH / 2
        _rounded_box(ax, left, badge_y - BADGE_HEIGHT / 2, BADGE_WIDTH, BADGE_HEIGHT, theme["card"],
                     radius=0.12, zorder=7, alpha=CARD_ALPHA)
        ax.text(left + 0.12, badge_y + 0.2, side, fontsize=8, fontweight="bold", color=theme["ink"],
                va="center", zorder=8)
        ax.text(left + BADGE_WIDTH - 0.12, badge_y + 0.2, f"{count}/{env.max_traffic_count}", fontsize=8,
                fontweight="bold", color=theme["ink"], ha="right", va="center", zorder=8)
        _progress_bar(ax, theme, left + 0.12, badge_y - 0.06, BADGE_WIDTH - 0.24,
                      count / env.max_traffic_count, height=0.09)
        ax.text(badge_x, badge_y - 0.24, state, fontsize=5.5, color=theme["ink_secondary"],
                ha="center", va="center", zorder=8)


def _draw_progress_card(ax, theme, env, x, y):
    inner_x, inner_width = x + CARD_PADDING, CARD_WIDTH - 2 * CARD_PADDING
    _card(ax, theme, x, y)
    _key_value(ax, theme, inner_x, y + CARD_HEIGHT - 0.3, inner_width, "STEP",
               f"{env.step_count} / {env.max_steps}")
    _progress_bar(ax, theme, inner_x, y + CARD_HEIGHT - 0.62, inner_width, env.step_count / env.max_steps)
    _key_value(ax, theme, inner_x, y + 0.6, inner_width, "CARS CLEARED",
               f"{env.cars_processed} / {env.max_cars_to_process}")
    _progress_bar(ax, theme, inner_x, y + 0.28, inner_width, env.cars_processed / env.max_cars_to_process)


def _draw_reward_card(ax, theme, env, x, y):
    inner_x, inner_width = x + CARD_PADDING, CARD_WIDTH - 2 * CARD_PADDING
    _card(ax, theme, x, y)

    rewards = env.reward_history
    last = "--" if env.last_reward is None else f"{env.last_reward:+.1f}"
    _key_value(ax, theme, inner_x, y + CARD_HEIGHT - 0.3, inner_width, "REWARD", last, fontsize=10)

    total = sum(rewards)
    average = total / len(rewards) if rewards else 0.0
    ax.text(inner_x, y + 0.18, f"total {total:+,.0f}", fontsize=6.5, color=theme["ink_secondary"], zorder=8)
    ax.text(inner_x + inner_width, y + 0.18, f"avg {average:+.1f}/step", fontsize=6.5,
            color=theme["ink_secondary"], ha="right", zorder=8)

    # Line of the last REWARD_WINDOW rewards. The x scale is fixed to the window, y fits the window and 0.
    window = rewards[-REWARD_WINDOW:]
    if len(window) > 1:
        left, right = inner_x, inner_x + inner_width
        bottom, top = y + 0.48, y + CARD_HEIGHT - 0.55
        low, high = min(min(window), 0), max(max(window), 0)
        span = (high - low) or 1

        xs = left + (right - left) * np.arange(len(window)) / (REWARD_WINDOW - 1)
        ys = bottom + (np.array(window) - low) / span * (top - bottom)
        zero = bottom + (0 - low) / span * (top - bottom)

        ax.plot([left, right], [zero, zero], color=theme["zero_line"], lw=1, zorder=7)
        ax.plot(xs, ys, color=theme["accent"], lw=1.4, zorder=8, solid_joinstyle="round")
        ax.add_patch(plt.Circle((xs[-1], ys[-1]), 0.06, facecolor=theme["accent"],
                                edgecolor=theme["card"], lw=1.5, zorder=9))
        ax.text(left, top + 0.04, f"last {REWARD_WINDOW}", fontsize=5.5, color=theme["ink_muted"],
                va="bottom", zorder=8)


def _draw_action_card(ax, theme, env, x, y):
    inner_x, inner_width = x + CARD_PADDING, CARD_WIDTH - 2 * CARD_PADDING
    _card(ax, theme, x, y)

    action = env.last_action.name if env.last_action is not None else "--"
    row_y = y + CARD_HEIGHT - 0.3
    if env.action_rejected:
        ax.text(inner_x, row_y, "ACTION", fontsize=6.5, color=theme["ink_muted"], va="center", zorder=8)
        ax.text(inner_x + inner_width, row_y, f"✕ {action} illegal", fontsize=7.5, color="#ffffff",
                fontweight="bold", ha="right", va="center", zorder=8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": CRITICAL, "edgecolor": "none"})
    else:
        _key_value(ax, theme, inner_x, row_y, inner_width, "ACTION", action)
    _key_value(ax, theme, inner_x, y + CARD_HEIGHT - 0.66, inner_width, "LIGHT", env.light_state.name)

    # One box per step before the reward decay starts, then the steps over and the decay multiplier
    green_time, threshold = env.current_time_in_gy, env.time_threshold
    over = max(0, green_time - threshold)
    _key_value(ax, theme, inner_x, y + 0.62, inner_width, "GREEN TIME",
               f"{green_time} steps" if green_time else "--")
    for i in range(threshold):
        color = theme["accent"] if i < green_time else theme["track"]
        _rounded_box(ax, inner_x + i * 0.34, y + 0.2, 0.28, 0.16, color, radius=0.04, zorder=8)
    if over:
        ax.text(inner_x + inner_width, y + 0.28,
                f"+{over} over  ·  reward ×{env.penalty_factor ** over:.2f}",
                fontsize=6.5, color=theme["ink"], ha="right", va="center", zorder=8)


def _draw_waiting_card(ax, theme, env, x, y):
    inner_x, inner_width = x + CARD_PADDING, CARD_WIDTH - 2 * CARD_PADDING
    _card(ax, theme, x, y)

    queued_waits = [car.waiting_time for queue in env.queues for car in queue]
    metrics = env.get_episode_metrics()
    _key_value(ax, theme, inner_x, y + CARD_HEIGHT - 0.3, inner_width, "WAITING ON RED",
               f"{env._get_cars_waiting()} cars")
    _key_value(ax, theme, inner_x, y + CARD_HEIGHT - 0.66, inner_width, "LONGEST QUEUED",
               f"{max(queued_waits, default=0)} steps")
    _key_value(ax, theme, inner_x, y + CARD_HEIGHT - 1.0, inner_width, "AVG WAIT (EPISODE)",
               f"{metrics['average_waiting_time']:.1f} steps", fontsize=8)

    # Legend of the car colors
    swatch_width = inner_width / len(WAIT_BINS)
    for i, color in enumerate(theme["cars"]):
        swatch_x = inner_x + i * swatch_width
        _rounded_box(ax, swatch_x, y + 0.3, swatch_width - 0.04, 0.14, color, radius=0.03, zorder=8)
        if i == len(WAIT_BINS) - 1:
            label = f"{WAIT_BINS[i]}+"
        else:
            label = f"{WAIT_BINS[i]}–{WAIT_BINS[i + 1] - 1}"
        ax.text(swatch_x + (swatch_width - 0.04) / 2, y + 0.16, label, fontsize=5.5,
                color=theme["ink_muted"], ha="center", va="center", zorder=8)


def draw_frame(env, theme_name="light"):
    """Draw the intersection and the episode information of `env`

    Args:
        env: TrafficLightEnv to draw
        theme_name: key of THEMES

    Returns:
        matplotlib.figure.Figure: the frame, FIGSIZE at DPI
    """
    theme = THEMES[theme_name]

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(theme["grass"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)
    ax.set_aspect("equal")
    ax.axis("off")

    _draw_roads(ax, theme)
    _draw_approaches(ax, theme, env)

    # Information cards in the four grass corners
    left, right = CARD_MARGIN, WORLD_SIZE - CARD_MARGIN - CARD_WIDTH
    bottom, top = CARD_MARGIN, WORLD_SIZE - CARD_MARGIN - CARD_HEIGHT
    _draw_progress_card(ax, theme, env, left, top)
    _draw_reward_card(ax, theme, env, right, top)
    _draw_action_card(ax, theme, env, left, bottom)
    _draw_waiting_card(ax, theme, env, right, bottom)

    if env.cars_processed >= env.max_cars_to_process:
        status = "TERMINATED: quota met"
    elif env.step_count >= env.max_steps:
        status = "TRUNCATED: time limit"
    else:
        status = None
    if status:
        ax.text(WORLD_SIZE / 2, WORLD_SIZE / 2, status, ha="center", va="center", fontsize=10,
                fontweight="bold", color=theme["ink"], zorder=10,
                bbox={"boxstyle": "round,pad=0.5", "facecolor": theme["card"], "edgecolor": "none"})

    return fig
