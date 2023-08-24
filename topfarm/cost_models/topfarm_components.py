from openmdao.core.explicitcomponent import ExplicitComponent
# from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
# from topfarm.constraint_components.post_constraint import PostConstraint
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
        for c in self.constraints:
            if not isinstance(c, ConstraintViolationComponent):
                self.add_input(c[0], val=c[1][next(iter(c[1]))])
        self.add_input('cost', val=0)
        self.add_output('final_cost', val=0)

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

    def compute(self, inputs, outputs):
        constraint_violations = []
        if 'constraint_violation' in inputs:
            constraint_violations.append(inputs['constraint_violation'])
        for c in self.constraints:
            if not isinstance(c, ConstraintViolationComponent):
                constraint_violations.append(self._get_constraint_violation(c, inputs[c[0]]))
        total_constraint_violation = sum(constraint_violations)
        if total_constraint_violation > 0:
            outputs['final_cost'] = total_constraint_violation + 1e10
        else:
            outputs['final_cost'] = inputs['cost']


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


# class PostPenaltyComponent(ExplicitComponent):
#     def __init__(self, post_constraint_lst, as_penalty):
#         super().__init__()
#         self.post_constraint_lst = post_constraint_lst
#         self.as_penalty = as_penalty

#     def setup(self):
#         for comp in self.post_constraint_lst:
#             if isinstance(comp[1], dict):
#                 # for key in comp[1]:
#                 self.add_input(comp[0], val=comp[1][next(iter(comp[1]))])
#             elif isinstance(comp, PostConstraint):
#                 self.add_input(comp.key, val=comp.upper)
#             else:
#                 self.add_input(comp[0], val=np.zeros(max([len(np.atleast_1d(c)) for c in comp[1:]])))
#         self.add_output('post_penalty', val=0.0)

#     def compute(self, inputs, outputs):
#         def get_penalty(post_constraint):
#             key = post_constraint[0]
#             if key.startswith('post_penalty'):
#                 return inputs[key]
#             else:
#                 pen = 0
#                 if isinstance(post_constraint[1], dict):
#                     for comp_key in post_constraint[1]:
#                         # inp_key = str(post_constraint[0] + '_' + comp_key)
#                         if comp_key == 'lower':
#                             pen = max(pen, np.max(post_constraint[1]['lower'] - inputs[key]))
#                         if comp_key == 'upper':
#                             pen = max(pen, inputs[key] + np.max(post_constraint[1]['upper']))
#                 else:
#                     warnings.warn("""constraint tuples should be of type (keyword, {constraint options}).""",
#                                       DeprecationWarning, stacklevel=2)
#                     lower, upper = post_constraint[1:]
#                     if lower is not None:
#                         pen = max(pen, np.max(lower - inputs[key]))
#                     if upper is not None:
#                         pen = max(pen, np.max(inputs[key] - upper))
#                 return pen
#         if self.as_penalty:
#             penalty = np.sum([get_penalty(pc) for pc in self.post_constraint_lst])
#         else:
#             penalty = 0.0
#         outputs['post_penalty'] = penalty
