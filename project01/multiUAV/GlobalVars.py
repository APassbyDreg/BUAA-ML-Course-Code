import numpy as np
from datetime import datetime

eps = 1e-6

# x, y, radius
__ENEMIES_RAW = [
    0.8750,    0.0500,    0.0300,
    0.4250,    0.1000,    0.0450,
    0.3000,    0.2000,    0.0500,
    0.4350,    0.2750,    0.0500,
    0.9250,    0.3000,    0.0500,
    0.6000,    0.5000,    0.0550,
    0.6750,    0.6500,    0.0250,
    0.8500,    0.6500,    0.0250,
    0.7500,    0.8000,    0.0625,
    0.5500,    0.8250,    0.0500,
]

# x, y , radius, reward point
__REWARDS_RAW = [
    0.7, 0.2, 0.08, 20,
    0.4, 0.5, 0.08, 20,
    0.1, 0.75, 0.1, 10,
    0.8, 0.7, 0.06, 30
]

# init_x, init_y, target_x, target_y
__UAVS_RAW = [
    0.3, 0.05, 0.9, 0.7,
    0.05, 0.6, 0.8, 0.4,
    0.6, 0.15, 0.1, 0.8
]

ALL_ENEMIES = np.array(__ENEMIES_RAW).reshape(len(__ENEMIES_RAW)//3, 3)
ENEMIES_UNKNOWN = ALL_ENEMIES.tolist()
REWARDS = np.array(__REWARDS_RAW).reshape(len(__REWARDS_RAW)//4, 4).tolist()
UAVS = np.array(__UAVS_RAW).reshape(len(__UAVS_RAW)//4, 2, 2).tolist()

# effect how large a step can be
STEP_SEGMENT = 20

# effect how detailed the integral process is
INTEGRAL_SEGMENT = 50

# effect danger map generation
ITER_COUNT = int(20 * STEP_SEGMENT)

# effect how danger map on iteration
DANGER_FACTOR = 10

# effect how threat decay over distance
# currently unused
THREAT_DECAY = 4

# effect enemy's radius on threat map
ENEMY_RADIUS_SPREAD = 1

# effects how threat map effect reward map (this is a power scale contol)
TR_COEFFICIENT = 5

# effect how uavs are willing to goto rewards (this is a power scale contol)
REWARD_SCALE = -5

# effect how fast reward point is decreasing through distance
REWARD_POINT_DECAY = 5

# threat radius when seen uav as a enemy
UAV_THREAT_RADIUS = (1 / STEP_SEGMENT) * 0.6

# result path
FIG_NAME_PREFIX = "fig"
RESULT_PATH = ".\\results\\{}".format(datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))