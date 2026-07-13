# Cartesian position reference for one robot frame, sampled on time_grid.
# x/y are expressed in the gait frame; z is absolute above ground.
string frame_name          # pinocchio frame name, e.g. "foot_l_contact", "pelvis"
bool[3] active_xyz
float64[3] weights_xyz
float64[] x                # len == len(time_grid)
float64[] y
float64[] z
