from openmdao.api import ExplicitComponent


class AggregatedCost(ExplicitComponent):
    '''
    Component that collects the cost and penalties and evaluates the aggregated cost
    '''
    def __init__(self, constraints_as_penalty, constraints, post_constraints):
        super().__init__()
        self.constraints_as_penalty = constraints_as_penalty
        self.constraints = constraints
        self.post_constraints = post_constraints

    def setup(self):
        self.add_input('cost', val=0)
        if self.constraints_as_penalty:
            if len(self.constraints) > 0:
                self.add_input('penalty', val=0)
            if len(self.post_constraints) > 0:
                self.add_input('post_penalty', val=0)
        self.add_output('aggr_cost', val=0)
        self.declare_partials(['*'], ['*'], method='fd')

    def compute(self, inputs, outputs):
        pen = 0
        if 'penalty' in inputs:
            pen += inputs['penalty']
        if 'post_penalty' in inputs:
            pen += inputs['post_penalty']
        if self.constraints_as_penalty and pen > 0:
            outputs['aggr_cost'] = pen + 10**10
        else:
            outputs['aggr_cost'] = inputs['cost']
