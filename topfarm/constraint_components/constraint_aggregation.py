# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:55:21 2022

@author: mikf
"""
from topfarm.constraint_components import Constraint, ConstraintComponent
from topfarm.cost_models.cost_model_wrappers import CostModelComponent


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
        self.constraint_aggregation_comp = ConstraintAggregationComp(**self.kwargs['component_args'])
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
    def __init__(self, **kwargs):
        CostModelComponent.__init__(self, **kwargs)

    def satisfy(self, state):
        pass
