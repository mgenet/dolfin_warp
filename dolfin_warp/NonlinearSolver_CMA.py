#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
### And Felipe Álvarez Barrientos, 2020-2026                                 ###
###                                                                          ###
### Pontificia Universidad Católica de Chile, Santiago, Chile                ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import numpy

try:
    import cma
except ImportError:
    pass

from .NonlinearSolver import NonlinearSolver

################################################################################

class CMANonlinearSolver(NonlinearSolver):



    def __init__(self,
            problem,
            parameters={}):

        self.problem    = problem
        self.printer    = self.problem.printer
        self.parameters = parameters

        self.working_folder   = self.parameters.get("working_folder"  )
        self.working_basename = self.parameters.get("working_basename")

        self.x_real_ini = numpy.asarray(self.parameters.get("x_real_ini", [ 0.0] * len(self.problem.reduced_displacement.vector().get_local())))
        self.x_real_min = numpy.asarray(self.parameters.get("x_real_min", [-0.1] * len(self.problem.reduced_displacement.vector().get_local())))
        self.x_real_max = numpy.asarray(self.parameters.get("x_real_max", [+0.3] * len(self.problem.reduced_displacement.vector().get_local())))

        self.problem.reduced_displacement.vector()[:] = self.x_real_ini

        self.x_norm_min = self.parameters.get("x_norm_min",  0.)
        self.x_norm_max = self.parameters.get("x_norm_max", 10.)

        self.solver_type = "cma"



    def solve(self,
            k_frame=None):

        self.k_frame = k_frame
        self.printer.print_str("k_frame: "+str(k_frame))

        x_real = self.problem.reduced_displacement.vector().get_local()
        x_norm = self.real2norm(x_real)

        options = {
            "bounds"              : [self.x_norm_min, self.x_norm_max]                                     ,
            "ftarget"             : self.parameters.get("ftarget"            , 1e-4                       ),
            "tolfun"              : self.parameters.get("tolfun"             , 1e-4                       ),
            "verb_filenameprefix" : self.parameters.get("verb_filenameprefix", self.working_folder+"/cma/"),
            "verb_log"            : self.parameters.get("verb_log"           , 100                        ),
            "tolflatfitness"      : self.parameters.get("tolflatfitness"     , 20                         )}

        if (("popsize" in self.parameters) and (self.parameters.get("popsize") is not None)):
            options["popsize"] = self.parameters.get("popsize")

        res = cma.fmin(
            objective_function = self.compute_ener                ,
            x0                 = x_norm                           ,
            sigma0             = self.parameters.get("sigma0", 2.),
            options            = options                          )

        self.printer.print_var("xbest (norm)",res[0]                )
        self.printer.print_var("xbest (real)",self.norm2real(res[0]))
        self.printer.print_var("fbest"       ,res[1]                )

        success = True
        x_norm  = res[0]
        n_iter  = res[4]

        x_real = self.norm2real(x_norm)
        self.problem.reduced_displacement.vector()[:] = x_real
        self.problem.update_disp()

        return success, n_iter



    def compute_ener(self,
            x_norm):

        x_real = self.norm2real(x_norm)
        self.problem.reduced_displacement.vector()[:] = x_real
        self.problem.call_before_assembly()
        ener = self.problem.assemble_ener()
        return ener



    def norm2real(self, x_norm):

        return self.x_real_min + (x_norm - self.x_norm_min)/(self.x_norm_max - self.x_norm_min) * (self.x_real_max - self.x_real_min)

    def real2norm(self, x_real):

        return self.x_norm_min + (x_real - self.x_real_min)/(self.x_real_max - self.x_real_min) * (self.x_norm_max - self.x_norm_min)
