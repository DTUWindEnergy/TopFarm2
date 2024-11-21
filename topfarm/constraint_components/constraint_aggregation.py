# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:55:21 2022

@author: mikf
"""
import numpy as np
from topfarm.constraint_components import Constraint, ConstraintComponent
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import (
    XYBoundaryConstraint,
    MultiXYBoundaryConstraint,
)


class ConstraintAggregation(Constraint):
    def __init__(self, constraints, **kwargs):
        self.constraints = constraints
        self.const_id = 'constraint_aggregation_comp'
        self.kwargs = kwargs
        self.input_keys = list([(i, i[0])[isinstance(i, tuple)] for i in kwargs['component_args']['input_keys']])
        self.output_keys = list([(i, i[0])[isinstance(i, tuple)] for i in kwargs['component_args']['output_keys']])

    @property
    def constraintComponent(self):
        return self.constraint_aggregation_comp

    def _setup(self, problem):
        self.constraint_aggregation_comp = ConstraintAggregationComp(self.constraints, **self.kwargs['component_args'])
        for constraint in self.constraints:
            constraint._setup(problem)
        problem.model.add_subsystem(self.const_id, self.constraint_aggregation_comp,
                                    promotes=['*'])

    def setup_as_constraint(self, problem):
        self._setup(problem)
        problem.model.add_constraint(**self.kwargs['constraint_args'])

    def setup_as_penalty(self, problem):
        self._setup(problem)


class ConstraintAggregationComp(CostModelComponent, ConstraintComponent):
    def __init__(self, constraints, **kwargs):
        self.constraints = constraints
        CostModelComponent.__init__(self, **kwargs)

    def satisfy(self, state):
        pass

    def plot(self, ax):
        for constraint in self.constraints:
            constraint.constraintComponent.plot(ax)


class DistanceConstraintAggregation(ConstraintAggregation):
    """Aggregating the spacing and boundary distances constraints into a penalty function"""

    def __init__(self, constraints, n_wt, min_spacing_m, windTurbines, name='sgd_constraint', **kwargs):
        """Initializing spacing constraint aggregation class

        Parameters
        ----------
        constraints : list
            list of constraint components, common constraints are [SpacingConstraints] and [XYBoundaryConstraints]
        n_wt : int
            number of turbines
        min_spacing_m : float
            minimum inter turbine spacing in meters
        windTurbines : object
            pywake object for the wind turbine used
        """

        def constr_aggr_func(wtSeparationSquared, boundaryDistances, **kwargs):
            separation_constraint = wtSeparationSquared - (2 * windTurbines.diameter()) ** 2
            separation_constraint = separation_constraint[separation_constraint < 0]
            distance_constraint = boundaryDistances
            distance_constraint = distance_constraint[distance_constraint < 0]
            return np.sum(-1 * separation_constraint) + np.sum(distance_constraint ** 2)

        def constr_aggr_grad(wtSeparationSquared, boundaryDistances, **kwargs):
            separation_constraint = wtSeparationSquared - (2 * windTurbines.diameter()) ** 2
            J_separation = np.zeros_like(wtSeparationSquared)
            J_separation[np.where(separation_constraint < 0)] = -1
            J_distance = np.zeros_like(boundaryDistances)
            J_distance[np.where(boundaryDistances < 0)] = 2 * boundaryDistances[np.where(boundaryDistances < 0)]
            return [[J_separation], [J_distance]]

        input_keys = []
        for cons in constraints:
            if isinstance(cons, XYBoundaryConstraint) or isinstance(
                cons, MultiXYBoundaryConstraint
            ):
                comp = cons.get_comp(n_wt)
                zeros = comp.zeros
                input_keys.append(('boundaryDistances', zeros))
            elif isinstance(cons, SpacingConstraint):
                zeros = np.zeros(int(n_wt * (n_wt - 1) / 2))
                input_keys.append(('wtSeparationSquared', zeros))

        kwargs['component_args'] = {'input_keys': input_keys,
                                    'n_wt': n_wt,
                                    'cost_function': constr_aggr_func,
                                    'cost_gradient_function': constr_aggr_grad,
                                    'objective': False,
                                    'output_keys': [(name, 0)],
                                    'use_constraint_violation': False
                                    }
        kwargs['constraint_args'] = {'name': name, 'lower': 0}

        ConstraintAggregation.__init__(self, constraints, **kwargs)
