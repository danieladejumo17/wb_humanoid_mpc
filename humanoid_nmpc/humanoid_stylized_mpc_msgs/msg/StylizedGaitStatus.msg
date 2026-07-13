# Published by the stylized MPC node: command acknowledgement and the phase
# anchor needed to align GUI playback with solver time.
uint32 last_command_id
bool accepted
string message                       # validation error / info text
bool active                          # a stylized cycle is currently driving the MPC
float64 anchor_time                  # solver time of the current cycle start
float64 cycle_duration
