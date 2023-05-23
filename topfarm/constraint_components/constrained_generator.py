from openmdao.drivers.doe_generators import DOEGenerator, ListGenerator
import numpy as np
import copy
import sys


class ConstrainedDiscardGenerator(DOEGenerator):
    def __init__(self, generator):
        DOEGenerator.__init__(self)
        if isinstance(generator, list):
            generator = ListGenerator(generator)
        self.generator = generator

    def __call__(self, design_vars, model=None):
        constr = model._get_subsystem('constraint_group')
        xy_boundary_comp = constr._get_subsystem('xy_bound_comp')
        spacing_comp = constr._get_subsystem('spacing_comp')
        for xyz_tuple_lst in self.generator(design_vars, model=model):
            x, y = ([np.array(t[1]) for t in xyz_tuple_lst] + [0])[:2]
            if spacing_comp:
                dist = spacing_comp._compute(x, y)
                if np.any(dist < spacing_comp.min_spacing**2):
                    continue

            if xy_boundary_comp:
                dist_to_boundary = xy_boundary_comp.distances(x, y)
                if np.any(dist_to_boundary < 0):
                    continue
            yield xyz_tuple_lst


class ConstrainedGenerator(DOEGenerator):
    def __init__(self, generator, n_iter=1000, step_size=0.1, offset=0.5, verbose=False):
        DOEGenerator.__init__(self)
        self.n_iter = n_iter
        self.step_size = step_size
        self.offset = offset
        if isinstance(generator, list):
            generator = ListGenerator(generator)
        self.generator = generator
        self.verbose = verbose

    def __call__(self, design_vars, model):
        for design_var_tuple_lst in self.generator(design_vars, model=model):
            state = {k.replace("indeps.", ''): np.array(v) for k, v in design_var_tuple_lst}
            for _ in range(100):
                state_before = copy.deepcopy(state)
                for constr_comp in model.constraint_components:
                    constr_comp.satisfy(state)
                if all([np.all(state[k] == state_before[k]) for k in state_before.keys()]):
                    break
            xyz_tuple_lst = [("indeps." + k, v) for k, v in state.items()]
            yield xyz_tuple_lst


class ConstrainedXYZGenerator(ConstrainedGenerator):
    def __init__(self, generator, n_iter=1000, step_size=0.1, offset=0.5, verbose=False):
        super().__init__(generator, n_iter=n_iter, step_size=step_size, offset=offset, verbose=verbose)
        sys.stderr.write("%s is deprecated. Use ConstrainedGenerator instead\n" % self.__class__.__name__)


class ConstrainedDiscardXYZGenerator(ConstrainedDiscardGenerator):
    def __init__(self, generator):
        ConstrainedDiscardGenerator.__init__(self, generator)
        sys.stderr.write("%s is deprecated. Use ConstrainedDiscardGenerator instead\n" % self.__class__.__name__)
