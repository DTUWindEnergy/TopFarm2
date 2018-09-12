from openmdao.drivers.doe_generators import DOEGenerator, ListGenerator
import numpy as np


class ConstrainedDiscardXYZGenerator(DOEGenerator):
    def __init__(self, generator):
        DOEGenerator.__init__(self)
        if isinstance(generator, list):
            generator = ListGenerator(generator)
        self.generator = generator

    def __call__(self, design_vars, model=None):
        xy_boundary_comp = model._get_subsystem('xy_bound_comp')
        spacing_comp = model._get_subsystem('spacing_comp')
        for xyz_tuple_lst in self.generator(design_vars, model=model):
            x, y, z = ([np.array(t[1]) for t in xyz_tuple_lst] + [0])[:3]
            if spacing_comp:
                dist = spacing_comp._compute(x, y)
                if np.any(dist < spacing_comp.min_spacing**2):
                    continue

            if xy_boundary_comp:
                dist_to_boundary = xy_boundary_comp.distances(x, y)
                if np.any(dist_to_boundary < 0):
                    continue
            yield xyz_tuple_lst


class ConstrainedXYZGenerator(DOEGenerator):
    def __init__(self, generator, n_iter=1000, step_size=0.1, offset=0.5, verbose=False):
        DOEGenerator.__init__(self)
        self.n_iter = n_iter
        self.step_size = step_size
        self.offset = offset
        if isinstance(generator, list):
            generator = ListGenerator(generator)
        self.generator = generator
        self.verbose = verbose

    def __call__(self, design_vars, model=None):
        xy_boundary_comp = model._get_subsystem('xy_bound_comp')
        spacing_comp = model._get_subsystem('spacing_comp')
        for xyz_tuple_lst in self.generator(design_vars, model=model):
            x, y, z = ([np.array(t[1]).astype(np.float) for t in xyz_tuple_lst] + [0])[:3]
            if spacing_comp:
                for j in range(self.n_iter):
                    dist = spacing_comp._compute(x, y)
                    dx, dy = spacing_comp._compute_partials(x, y)
                    index = int(np.argmin(dist))
                    if dist[index] < spacing_comp.min_spacing**2 or j == 0:
                        x += dx[index] * self.step_size
                        y += dy[index] * self.step_size
                        if xy_boundary_comp:
                            x, y, z = xy_boundary_comp.move_inside(x, y, z)
                    else:
                        if self.verbose:
                            print('Obtained required spacing after %d iterations' % j)
                        xyz = np.array([x, y, z])
                        break

            elif xy_boundary_comp:
                xyz = xy_boundary_comp.move_inside(x, y, z)
            xyz_tuple_lst = [(t[0], xyz) for t, xyz in zip(xyz_tuple_lst, xyz)]
            yield xyz_tuple_lst
#
#
# class shuffle_generator(DOEGenerator):
#     def __init__(self, turbineXYZOptimizationProblem, N=10, shuffle_type='rel', n_iter=1000,
#                  step_size=0.1, pad=1.1, offset=5, plot=False,
#                  verbose=False):
#         self.N = N
#         self.shuffle_type = shuffle_type
#         self.n_iter = n_iter
#         self.step_size = step_size
#         self.pad = pad
#         self.offset = offset
#         self.plot = plot
#         self.verbose = verbose
#         DOEGenerator.__init__(self)
#         self.turbineXYZOptimizationProblem = turbineXYZOptimizationProblem
#
#     def __call__(self, *args, **kwargs):
#         for _ in range(self.N):
#             p = self.turbineXYZOptimizationProblem
#             turbines = spos(p.xy_boundary, p.n_wt, p.min_spacing,
#                             p.turbine_positions, self.shuffle_type, self.n_iter,
#                             self.step_size, self.pad, self.offset, self.plot, self.verbose)
#             x, y = turbines.T
#             yield np.array([x, y, self.initialXYZOptimizationProblem['turbineZ']])
