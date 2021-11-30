from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
from topfarm.constraint_components.post_constraint import PostConstraint
import numpy as np


class PenaltyComponent(ExplicitComponent):
    def __init__(self, constraint_comps, as_penalty):
        super().__init__()
        self.post_constraint_lst = constraint_comps
        self.as_penalty = as_penalty

    def setup(self):
        for comp in self.post_constraint_lst:
            self.add_input('penalty_' + comp.const_id, val=0.0)
        self.add_output('penalty', val=0.0)

    def compute(self, inputs, outputs):
        penalties = np.array([inputs[x] for x in inputs])
        if self.as_penalty:
            penalty = sum(penalties)
        else:
            penalty = 0.0
        outputs['penalty'] = penalty


class PostPenaltyComponent(ExplicitComponent):
    def __init__(self, post_constraint_lst, as_penalty):
        super().__init__()
        self.post_constraint_lst = post_constraint_lst
        self.as_penalty = as_penalty

    def setup(self):
        for comp in self.post_constraint_lst:
            if isinstance(comp[1], dict):
                # for key in comp[1]:
                self.add_input(comp[0], val=comp[1][next(iter(comp[1]))])
            elif isinstance(comp, PostConstraint):
                self.add_input(comp.key, val=comp.upper)
            else:
                self.add_input(comp[0], val=np.zeros(max([len(np.atleast_1d(c)) for c in comp[1:]])))
        self.add_output('post_penalty', val=0.0)

    def compute(self, inputs, outputs):
        def get_penalty(post_constraint):
            key = post_constraint[0]
            if key.startswith('post_penalty'):
                return inputs[key]
            else:
                pen = 0
                if isinstance(post_constraint[1], dict):
                    for comp_key in post_constraint[1]:
                        # inp_key = str(post_constraint[0] + '_' + comp_key)
                        if comp_key == 'lower':
                            pen = max(pen, np.max(post_constraint[1]['lower'] - inputs[key]))
                        if comp_key == 'upper':
                            pen = max(pen, inputs[key] + np.max(post_constraint[1]['upper']))
                else:
                    lower, upper = post_constraint[1:]
                    if lower is not None:
                        pen = max(pen, np.max(lower - inputs[key]))
                    if upper is not None:
                        pen = max(pen, np.max(inputs[key] - upper))
                return pen
        if self.as_penalty:
            penalty = np.sum([get_penalty(pc) for pc in self.post_constraint_lst])
        else:
            penalty = 0.0
        outputs['post_penalty'] = penalty
