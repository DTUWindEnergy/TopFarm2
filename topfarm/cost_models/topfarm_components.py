from openmdao.core.explicitcomponent import ExplicitComponent
import numpy as np
import warnings
from topfarm.constraint_components.boundary import BoundaryBaseComp
from topfarm.constraint_components.spacing import SpacingComp, SpacingTypeComp
from topfarm.constraint_components.capacity import CapacityComp


class ConstraintViolationComponent(ExplicitComponent):
    def __init__(self, constraint_comps):
        super().__init__()
        self.constraint_comps = constraint_comps

    def setup(self):
        for comp in self.constraint_comps:
            if isinstance(comp.constraintComponent, BoundaryBaseComp):
                self.add_input('boundaryDistances', val=comp.constraintComponent.zeros)
                self.declare_partials('constraint_violation', 'boundaryDistances')
            elif isinstance(comp.constraintComponent, SpacingTypeComp):
                self.add_input('wtRelativeSeparationSquared', val=np.zeros(comp.constraintComponent.veclen))
                self.declare_partials('constraint_violation', 'wtRelativeSeparationSquared')
            elif isinstance(comp.constraintComponent, SpacingComp):
                self.add_input('wtSeparationSquared', val=np.zeros(comp.constraintComponent.veclen))
                self.declare_partials('constraint_violation', 'wtSeparationSquared')
            elif isinstance(comp.constraintComponent, CapacityComp):
                self.add_input('totalcapacity', val=0.0)
                self.declare_partials('constraint_violation', 'totalcapacity')
        self.add_output('constraint_violation', val=0.0)

    def compute(self, inputs, outputs):
        constraint_violations = []
        for comp in self.constraint_comps:
            if isinstance(comp.constraintComponent, BoundaryBaseComp):
                constraint_violations.append(np.sum(np.minimum(inputs['boundaryDistances'], 0) ** 2))
            elif isinstance(comp.constraintComponent, SpacingTypeComp):
                constraint_violations.append(-np.minimum(inputs['wtRelativeSeparationSquared'], 0).sum())
            elif isinstance(comp.constraintComponent, SpacingComp):
                constraint_violations.append(-np.minimum(inputs['wtSeparationSquared'] - comp.constraintComponent.min_spacing ** 2, 0).sum())
            elif isinstance(comp.constraintComponent, CapacityComp):
                constraint_violations.append(np.maximum(0, inputs['totalcapacity'][0] - comp.constraintComponent.max_capacity))
        outputs['constraint_violation'] = sum(constraint_violations)

    def compute_partials(self, inputs, partials):
        for comp in self.constraint_comps:
            if isinstance(comp.constraintComponent, BoundaryBaseComp):
                dcdb = np.zeros_like(inputs['boundaryDistances'])
                index = np.where(inputs['boundaryDistances'] < 0)
                dcdb[index] = 2 * inputs['boundaryDistances'][index]
                partials['constraint_violation', 'boundaryDistances'] = dcdb
            elif isinstance(comp.constraintComponent, SpacingTypeComp):
                dcdsr = np.zeros_like(inputs['wtRelativeSeparationSquared'])
                index = np.where(inputs['wtRelativeSeparationSquared'] < 0)
                dcdsr[index] = -1
                partials['constraint_violation', 'wtRelativeSeparationSquared'] = dcdsr
            elif isinstance(comp.constraintComponent, SpacingComp):
                dcds = np.zeros_like(inputs['wtSeparationSquared'])
                index = np.where(inputs['wtSeparationSquared'] - comp.constraintComponent.min_spacing ** 2 < 0)
                dcds[index] = -1
                partials['constraint_violation', 'wtSeparationSquared'] = dcds
            elif isinstance(comp.constraintComponent, CapacityComp):
                dcds = np.zeros_like(inputs['totalcapacity'])
                index = np.where(inputs['totalcapacity'] - comp.constraintComponent.max_capacity > 0)
                dcds[index] = 1
                partials['constraint_violation', 'totalcapacity'] = dcds


class ObjectiveComponent(ExplicitComponent):
    def __init__(self, constraints):
        super().__init__()
        self.constraints = constraints

    def setup(self):
        if np.any([isinstance(c, ConstraintViolationComponent) for c in self.constraints]):
            self.add_input('constraint_violation')
            self.declare_partials('final_cost', 'constraint_violation', val=0, method='fd')
        for c in self.constraints:
            if not isinstance(c, ConstraintViolationComponent):
                self.add_input(c[0], val=c[1][next(iter(c[1]))])
        self.add_input('cost', val=0)
        self.add_output('final_cost', val=0)
        self.declare_partials('final_cost', 'cost', val=1)

    def _get_constraint_violation(self, constraint, constraint_val):
        upper = [0]
        lower = [0]
        if isinstance(constraint[1], dict):
            constraint_args = constraint[1]
            if 'upper' in constraint_args:
                upper = constraint_val - constraint_args['upper']
            if 'lower' in constraint_args:
                lower = - constraint_val + constraint_args['lower']
        else:
            warnings.warn("constraint tuple should be of type (constraint_key, {constraint_args})")
        return max(0, *upper, *lower)

    def _compute(self, inputs):
        constraint_violations = []
        if 'constraint_violation' in inputs:
            constraint_violations.append(inputs['constraint_violation'])
        for c in self.constraints:
            if not isinstance(c, ConstraintViolationComponent):
                constraint_violations.append(self._get_constraint_violation(c, inputs[c[0]]))
        return sum(constraint_violations)

    def compute(self, inputs, outputs):
        total_constraint_violation = self._compute(inputs)
        if total_constraint_violation > 0:
            outputs['final_cost'] = 1e10 + total_constraint_violation
        else:
            outputs['final_cost'] = inputs['cost']

    def compute_partials(self, inputs, J):
        total_constraint_violation = self._compute(inputs)
        if total_constraint_violation > 0:
            J['final_cost', 'cost'] = 0
        else:
            J['final_cost', 'cost'] = 1


class DummyObjectiveComponent(ExplicitComponent):
    def __init__(self):
        super().__init__()

    def setup(self):
        self.add_input('cost', val=0)
        self.add_output('final_cost', val=0)
        self.declare_partials('final_cost', 'cost', val=1)

    def compute(self, inputs, outputs):
        outputs['final_cost'] = inputs['cost']

    def compute_partials(self, inputs, J):
        J['final_cost', 'cost'] = 1
