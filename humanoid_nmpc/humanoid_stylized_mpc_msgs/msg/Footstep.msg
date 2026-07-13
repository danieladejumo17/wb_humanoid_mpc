# One planted foot position within the gait cycle (gait-frame XY).
# touchdown_time == 0.0 marks the initial stance position at cycle start.
int8 FOOT_LEFT=0
int8 FOOT_RIGHT=1

int8 foot
float64 touchdown_time     # in [0, cycle_duration]
float64[2] position_xy
