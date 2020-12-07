import os 

from UAVControls import *

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

os.system("copy *.py {}".format(RESULT_PATH[2:]))

group = UAVGroup()

for uav_setup in UAVS:
    group.addUAV(uav_setup[0], uav_setup[1])

while not group.ended:
    group.moveOnce()
