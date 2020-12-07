import numpy as np

from Utils import *

class UAV:
    """
        a single UAV
    """
    # initials
    init_pos = None
    # basic properties
    idx = 0
    curr_pos = np.array([0, 0])
    target = np.array([1, 1])
    obs_radius = 0.2
    history = []
    # trace related
    next_step = None

    @property
    def reached(self):
        return np.linalg.norm(self.curr_pos - self.target) < 1 / INTEGRAL_SEGMENT

    def __init__(self, idx, init_pos, target, obs_radius):
        self.idx = idx
        self.curr_pos = np.array(init_pos)
        self.init_pos = np.array(init_pos)
        self.target = np.array(target)
        self.obs_radius = obs_radius
        self.history = [self.curr_pos.tolist()]

    def plan(self, threat_map, reward_map=None ,scaler=INTEGRAL_SEGMENT):
        # plan next step (because program will replan after every step move)
        if reward_map is None:
            reward_map = np.zeros([scaler, scaler])
        else:
            reward_map = rewardMapAtPos(self.curr_pos, reward_map)
        # create weights map
        choice_weights = - reward_map / np.exp(threat_map * TR_COEFFICIENT)
        choice_weights += calcDangerMatrix(np.copy(self.curr_pos), threat_map, n_iter=1)
        choice_weights += calcDangerMatrix(np.copy(self.target), threat_map)
        # scale curr pos to avoid self min
        choice_weights[floor(self.curr_pos[0] * scaler), floor(self.curr_pos[1] * scaler)] *= 10
        # choose the min point of weights map as next point
        self.next_step = np.array(np.unravel_index(choice_weights.argmin(), choice_weights.shape)) / INTEGRAL_SEGMENT

    def stepOnce(self):
        direction = self.next_step - self.curr_pos
        dir_norm = np.linalg.norm(direction)
        if dir_norm <= 1 / STEP_SEGMENT:
            self.curr_pos = self.next_step
        else:
            self.curr_pos += direction / dir_norm * (1 / STEP_SEGMENT)
        self.curr_pos = np.floor(self.curr_pos * INTEGRAL_SEGMENT) / INTEGRAL_SEGMENT
        self.history.append(self.curr_pos.tolist())
        print("uav {} from {} to {}".format(self.idx, self.history[-2], self.history[-1]))


class UAVGroup:
    """
        group of UAVs that have communications
    """
    uavs = []
    known_threats = np.zeros([INTEGRAL_SEGMENT, INTEGRAL_SEGMENT])
    remain_rewards = None
    moves_taken = 0

    @property
    def ended(self):
        for uav in self.uavs:
            if not uav.reached:
                return False
        return True

    def __init__(self):
        self.remain_rewards = updateRewardMap(REWARDS)

    def positions_as_enemies(self):
        pos = []
        for uav in self.uavs:
            pos.append(uav.curr_pos.tolist() + [UAV_THREAT_RADIUS])
        return pos

    def addUAV(self, init_pos, target, obs_radius=2/STEP_SEGMENT):
        self.uavs.append(UAV(len(self.uavs), init_pos, target, obs_radius))

    def moveOnce(self):
        saveStatusFig(self)
        # update info
        uav_as_enemy = self.positions_as_enemies()
        new_enemies = updateKnownEnemies(self.uavs)
        new_rewards = updateDoneRewards(self.uavs)
        self.remain_rewards = updateRewardMap(removed=new_rewards, previous=self.remain_rewards)
        self.known_threats = calcEnemyThreat(new_enemies, pre_calculated=self.known_threats, source="ENEMY")
        # plan
        for uav in self.uavs:
            if not uav.reached:
                other_uavs = uav_as_enemy[:uav.idx] + uav_as_enemy[uav.idx+1:]
                all_threat = calcEnemyThreat(other_uavs, self.known_threats, source="UAV")
                uav.plan(all_threat, self.remain_rewards)
        # move
        for uav in self.uavs:
            if not uav.reached:
                uav.stepOnce()
        # save status fig at exit
        self.moves_taken += 1
        if self.ended:
            saveStatusFig(self)
