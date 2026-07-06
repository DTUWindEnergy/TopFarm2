"""Shared constants used across TopFarm."""

# Design-variable string keys
x_key = "x"
y_key = "y"
z_key = "z"
type_key = "type"
cost_key = "cost"
grid_x_key = "sx"
grid_y_key = "sy"

# Penalty added to the objective when a constraint is violated (used by
# drivers that do not support native constraints, e.g. SimpleGADriver).
CONSTRAINT_PENALTY = 1e10
