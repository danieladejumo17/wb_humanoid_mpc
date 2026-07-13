# One scalar reference curve, sampled on StylizedGaitCommand/time_grid.
# name: "joint/<mpc_joint_name>" or "base/{px,py,pz,yaw,pitch,roll}".
# Base px/py/yaw are relative to the gait frame (cycle-start base anchor);
# pz is absolute height above ground; pitch/roll absolute.
string name
bool active
float64 weight        # tracking weight; 0 => use the task.info default
float64[] values      # len == len(time_grid)
