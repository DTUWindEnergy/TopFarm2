# from openmdao.core.explicitcomponent import ExplicitComponent
# import numpy as np
# import warnings


# class ConstraintViolationComponent(ExplicitComponent):
#     def __init__(self, constraint_comps):
#         super().__init__()
#         self.constraint_comps = constraint_comps

#     def setup(self):
#         for comp in self.constraint_comps:
#             self.add_input('constraint_violation_' + comp.const_id, val=0.0)
#         self.add_output('constraint_violation', val=0.0)

#     def compute(self, inputs, outputs):
#         constraint_violations = np.array([inputs[x] for x in inputs])
#         outputs['constraint_violation'] = sum(constraint_violations)


# class ObjectiveComponent(ExplicitComponent):
#     def __init__(self, constraints):
#         super().__init__()
#         self.constraints = constraints

#     def setup(self):
#         if np.any([isinstance(c, ConstraintViolationComponent) for c in self.constraints]):
#             self.add_input('constraint_violation')
#         for c in self.constraints:
#             if not isinstance(c, ConstraintViolationComponent):
#                 self.add_input(c[0], val=c[1][next(iter(c[1]))])
#         self.add_input('cost', val=0)
#         self.add_output('final_cost', val=0)

#     def _get_constraint_violation(self, constraint, constraint_val):
#         upper = 0
#         lower = 0
#         if isinstance(constraint[1], dict):
#             constraint_args = constraint[1]
#             if 'upper' in constraint_args:
#                 upper = constraint_val - constraint_args['upper']
#             if 'lower' in constraint_args:
#                 lower = - constraint_val + constraint_args['lower']
#         else:
#             warnings.warn("constraint tuple should be of type (constraint_key, {constraint_args})")
#         return max(0, upper, lower)

#     def compute(self, inputs, outputs):
#         constraint_violations = []
#         if 'constraint_violation' in inputs:
#             constraint_violations.append(inputs['constraint_violation'])
#         for c in self.constraints:
#             if not isinstance(c, ConstraintViolationComponent):
#                 constraint_violations.append(self._get_constraint_violation(c, inputs[c[0]]))
#         total_constraint_violation = sum(constraint_violations)
#         if total_constraint_violation > 0:
#             outputs['final_cost'] = total_constraint_violation + 1e10
#         else:
#             outputs['final_cost'] = inputs['cost']


# class DummyObjectiveComponent(ExplicitComponent):
#     def __init__(self):
#         super().__init__()

#     def setup(self):
#         self.add_input('cost', val=0)
#         self.add_output('final_cost', val=0)
#         self.declare_partials('final_cost', 'cost', val=1)

#     def compute(self, inputs, outputs):
#         self.outputs['final_cost'] = inputs['cost']

#     def compute_partials(self, inputs, J):
#         J['final_cost', 'cost'] = 1
