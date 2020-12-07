from math import floor, sqrt
import os

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.pyplot import hsv

from GlobalVars import *


def guassianThreat(enemy, source, scaler=INTEGRAL_SEGMENT):
    """
        return single threat map for one enemy
    """
    pos = np.array(enemy[:2])
    radius = enemy[2] 
    distances = np.zeros([scaler, scaler])
    for x in range(scaler):
        for y in range(scaler):
            # modifies variance and distance algorism
            if source == "ENEMY":
                distances[x, y] = max(0, np.linalg.norm(np.array([x,y])/scaler - pos) - radius)
            elif source == "UAV":
                distances[x, y] = np.linalg.norm(np.array([x, y])/scaler - pos)
            else:
                raise ValueError("invalid threat source")
            ## OLD: distances[x, y] = np.linalg.norm(np.array([x,y])/scaler - pos)
    return np.power(np.math.exp(1), - distances**2 / ((radius * ENEMY_RADIUS_SPREAD/2)**2))
    ## OLD 0: return np.power(np.math.exp(1), - distances**2 / 2*radius**2))


def calcEnemyThreat(enemies_added, pre_calculated=None, scaler=INTEGRAL_SEGMENT, source="ENEMY"):
    """
        return the guassian distribution of given enemies
        if possible, use pre_calculated ones to improve speed
    """
    if pre_calculated is None:
        all_threat = np.zeros([scaler, scaler])
    else:
        all_threat = np.copy(pre_calculated)
    # go through enemies
    for enemy in enemies_added:
        curr_threat = guassianThreat(enemy, source, scaler)
        all_threat = 1 - (1-all_threat) * (1-curr_threat)

    return all_threat


def calcDangerMatrix(target, threat_matrix, scaler=INTEGRAL_SEGMENT, n_iter=ITER_COUNT):
    """
        return danger matrix D after n_iter times of iteration
    """
    # init D
    target_scaled = [floor(axis * scaler) for axis in target]
    D = np.ones([scaler, scaler]) * (scaler**2)
    D[target_scaled[0], target_scaled[1]] = 0

    # iters
    curr_start = np.copy(target)
    integral_step = 1 / scaler
    for _ in range(n_iter):
        danger_to_target = np.zeros([scaler, scaler])
        for x in range(scaler):
            for y in range(scaler):
                pos = np.array([x, y]) / scaler
                dis = np.linalg.norm(pos - curr_start)
                danger_to_target[x, y] = dis
                # move through dis step by step
                delta = 0
                integral_pos = np.copy(curr_start)
                while dis > 0:
                    dis_move = min(integral_step, dis)
                    integral_pos += (pos - integral_pos) * (dis_move / dis)
                    delta += dis_move * threat_matrix[floor(integral_pos[0]*scaler), floor(integral_pos[1]*scaler)]
                    dis -= dis_move
                # add to danger to this target
                danger_to_target[x, y] += DANGER_FACTOR * delta
        # update D and curr_target
        danger_to_target += D[floor(curr_start[0]*scaler), floor(curr_start[1]*scaler)]
        D = np.where(D < danger_to_target, D, danger_to_target)     # D = min(D, A)
        curr_start = np.random.rand(2)

    return D


def updateKnownEnemies(uavs):
    """
        check unknown enemies and return new ones
    """
    updated = []
    for eidx in range(len(ENEMIES_UNKNOWN)):
        idx = eidx - len(updated)
        for uav in uavs:
            upos = uav.curr_pos
            distance = sqrt((upos[0] - ENEMIES_UNKNOWN[idx][0])**2 + (upos[1] - ENEMIES_UNKNOWN[idx][1])**2)
            if distance < uav.obs_radius + ENEMIES_UNKNOWN[idx][2]:
                updated.append(ENEMIES_UNKNOWN.pop(idx))
                break
    return updated


def updateDoneRewards(uavs):
    """
        check undown rewards and return done ones
    """
    updated = []
    for ridx in range(len(REWARDS)):
        idx = ridx - len(updated)
        for uav in uavs:
            upos = uav.curr_pos
            distance = sqrt((upos[0] - REWARDS[idx][0])**2 + (upos[1] - REWARDS[idx][1])**2)
            if distance < REWARDS[idx][2]:
                updated.append(REWARDS.pop(idx))
                break
    return updated


def updateRewardMap(added=[], removed=[], previous=None, scaler=INTEGRAL_SEGMENT):
    """
        update reward map with rewards added / removed
        utilizing previous result
    """
    if previous is None:
        reward = np.zeros([scaler, scaler])
    else:
        reward = previous
    for inc in added:
        pos = np.array(inc[:2])
        for x in range(scaler):
            for y in range(scaler):
                distance = np.linalg.norm(np.array([x, y])/scaler - pos)
                if distance < inc[2]:
                    reward[x, y] += inc[3] * np.exp(REWARD_SCALE)
    for dec in removed:
        pos = np.array(dec[:2])
        for x in range(scaler):
            for y in range(scaler):
                distance = np.linalg.norm(np.array([x, y])/scaler - pos)
                if distance < dec[2]:
                    reward[x, y] -= dec[3] * np.exp(REWARD_SCALE)
    return reward


def rewardMapAtPos(pos, original_map, scaler=INTEGRAL_SEGMENT):
    """
        specifies reward at given point
        (further -> less reward point)
    """
    reward = np.zeros(original_map.shape)
    for x in range(original_map.shape[0]):
        for y in range(original_map.shape[1]):
            distance = np.linalg.norm(np.array([x, y])/scaler - pos)
            reward[x][y] = ((max(0, 1-distance)) ** REWARD_POINT_DECAY) * original_map[x][y]
    return reward


def saveStatusFig(group):
    """
        save fig
    """
    plt.xlim(-0.25, 1.25)
    plt.ylim(-0.25, 1.25)
    fig, ax = plt.subplots()
    ax.set_aspect(1)
    # draw rewards and enemies
    for r in REWARDS:
        circle = plt.Circle(r[:2], r[2], alpha=0.2, color="blue")
        ax.add_artist(circle)
    for e in ALL_ENEMIES:
        circle = plt.Circle(e[:2], e[2], alpha=0.2, color="red")
        ax.add_artist(circle)
    # draw threats
    x = np.arange(0, 1, 1/INTEGRAL_SEGMENT)
    y = np.arange(0, 1, 1/INTEGRAL_SEGMENT)
    grid_x, grid_y = np.meshgrid(x, y)
    ax.contour(grid_x, grid_y,np.transpose(group.known_threats))
    # draw paths
    for uav in group.uavs:
        path = np.array(uav.history)
        color = hsv_to_rgb([0.17*uav.idx, 0.8, 0.9])
        ax.plot(path[:, 0], path[:, 1], label="uav{}".format(uav.idx), color=color)
        ax.scatter(uav.init_pos[0], uav.init_pos[1], marker="P", color=color)
        ax.scatter(uav.curr_pos[0], uav.curr_pos[1], marker="+", color=color)
        ax.scatter(uav.target[0], uav.target[1], marker="*", color=color)
    fig.savefig(os.path.join(RESULT_PATH,FIG_NAME_PREFIX+str(group.moves_taken)), dpi=250)
    plt.close(fig)
    
