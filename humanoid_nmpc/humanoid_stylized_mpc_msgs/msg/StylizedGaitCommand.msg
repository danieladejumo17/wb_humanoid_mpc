# A full stylized gait cycle description. The MPC tiles this cycle
# periodically over its horizon, advancing the gait-frame anchor by the
# per-cycle footstep displacement.
std_msgs/Header header
uint32 command_id
bool enable                          # false => fall back to standing

float64 cycle_duration               # seconds, > 0
float64[] time_grid                  # uniform samples over [0, cycle_duration], incl. endpoints

# Contact schedule of one cycle, template-style: event_times has
# len(mode_sequence)+1 entries and includes 0.0 and cycle_duration.
# Modes: FLY=0, RF=1, LF=2, STANCE=3 (MotionPhaseDefinition.h).
ocs2_ros2_msgs/ModeSchedule cycle_mode_schedule

CurveChannel[] joint_curves
CurveChannel[] base_curves
CartesianCurve[] cartesian_curves
Footstep[] footsteps

float64 footstep_weight              # weight of the touchdown-placement cost
float64[] swing_apex_heights         # optional per-swing apex override; empty = default
