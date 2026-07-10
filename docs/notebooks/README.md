# TOPFARM Notebooks

This folder contains Jupyter notebooks that demonstrate TOPFARM features.
Start with **Fundamentals** if you are new to TOPFARM.

## Fundamentals

| Notebook | Description |
|----------|-------------|
| [design_variables.ipynb](design_variables.ipynb) | How to define x/y/z positions and turbine types as design variables |
| [constraints.ipynb](constraints.ipynb) | Boundary and spacing constraints: convex hull, polygon, circle |
| [drivers.ipynb](drivers.ipynb) | Overview of available optimization drivers (SLSQP, GA, SGD, random search) |
| [problems.ipynb](problems.ipynb) | Constructing and running a `TopFarmProblem` end to end |

## Layout Optimization

| Notebook | Description |
|----------|-------------|
| [exclusion_zones.ipynb](exclusion_zones.ipynb) | Layouts with exclusion zones using multi-polygon boundaries |
| [turbine_specific_boundaries.ipynb](turbine_specific_boundaries.ipynb) | Per-turbine boundary constraints for heterogeneous sites |
| [neighbouring_turbines.ipynb](neighbouring_turbines.ipynb) | Including fixed neighbouring turbines outside the design domain |
| [regular_grid_optimization.ipynb](regular_grid_optimization.ipynb) | Optimizing row/column spacing on a regular grid |
| [smart_start_with_predefined_types.ipynb](smart_start_with_predefined_types.ipynb) | Data-driven initialization with Smart Start |

## Advanced Optimization

| Notebook | Description |
|----------|-------------|
| [nested_optimization_type_and_positions.ipynb](nested_optimization_type_and_positions.ipynb) | Simultaneously optimize layout and turbine types using nested problems |
| [hub_height_optimization.ipynb](hub_height_optimization.ipynb) | Add hub height as a design variable |
| [joint_multi_wf_optimization.ipynb](joint_multi_wf_optimization.ipynb) | Jointly optimize multiple wind farms |
| [wind_farm_cluster.ipynb](wind_farm_cluster.ipynb) | Cluster-level optimization across a group of farms |
| [sgd_slsqp_comparison.ipynb](sgd_slsqp_comparison.ipynb) | Compare stochastic gradient descent with SLSQP |
| [unconstrained_wake_steering.ipynb](unconstrained_wake_steering.ipynb) | Yaw angle optimization for wake steering |

## Loads & Economics

| Notebook | Description |
|----------|-------------|
| [layout_and_loads.ipynb](layout_and_loads.ipynb) | Co-optimize layout and structural loads |
| [wake_steering_and_loads.ipynb](wake_steering_and_loads.ipynb) | Wake steering subject to load constraints |
| [economic_cost_models.ipynb](economic_cost_models.ipynb) | CAPEX/OPEX cost models and NPV optimization |
| [cables.ipynb](cables.ipynb) | Cable layout cost as part of the objective |

## Site-Specific

| Notebook | Description |
|----------|-------------|
| [bathymetry.ipynb](bathymetry.ipynb) | Offshore layout with bathymetry-dependent foundation costs |
| [example_randomize_type_and_position_with_specific_boundaries.ipynb](example_randomize_type_and_position_with_specific_boundaries.ipynb) | Random-search driver with type and position randomization |

## Running the Notebooks

```bash
pip install topfarm jupyter
jupyter notebook
```

All notebooks assume PyWake is installed for wake modelling:

```bash
pip install py_wake
```
