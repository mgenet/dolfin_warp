#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
### And Felipe Álvarez Barrientos, 2020-2025                                 ###
###                                                                          ###
### Pontificia Universidad Católica de Chile, Santiago, Chile                ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import numpy

try:
    import cma
    has_cma = True
except ImportError:
    has_cma = False

try:
    import scipy.optimize
    has_scipy = True
except ImportError:
    has_scipy = False

from .NonlinearSolver import NonlinearSolver

################################################################################

class GradientFreeNonlinearSolver(NonlinearSolver):



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

        self.solver_type = self.parameters.get("solver_type", "cma")
        # self.solver_type = self.parameters.get("solver_type", "scipy-minimize")
        # self.solver_type = self.parameters.get("solver_type", "scipy-differential_evolution")



    def solve(self,
            k_frame=None):

        self.k_frame = k_frame
        self.printer.print_str("k_frame: "+str(k_frame))

        x_real = self.problem.reduced_displacement.vector().get_local()
        x_norm = self.real2norm(x_real)

        if (self.solver_type == "cma"):

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

        elif (self.solver_type.startswith("scipy-minimize")): # local solvers

            res = scipy.optimize.minimize(
                fun           = self.compute_ener                                 ,
                x0            = x_norm                                            ,
                method        = self.solver_type.split("-", 2)[2]                 ,
                bounds        = [(self.x_norm_min, self.x_norm_max)] * len(x_norm),
                tol           = self.parameters.get("tol", 1e-4)                  )

            self.printer.print_var("success" ,res.success          )
            self.printer.print_var("message" ,res.message          )
            self.printer.print_var("x (norm)",res.x                )
            self.printer.print_var("x (real)",self.norm2real(res.x))

            success = res.success
            x_norm  = res.x
            n_iter  = res.nit

        elif (self.solver_type == "scipy-differential_evolution"): # global solver

            res = scipy.optimize.differential_evolution(
                func          = self.compute_ener                                 ,
                bounds        = [(self.x_norm_min, self.x_norm_max)] * len(x_norm),
                strategy      = self.parameters.get("strategy"     , "best1bin")  , # best1bin or "rand1bin" if you want more exploration
                maxiter       = self.parameters.get("maxiter"      , 1000      )  ,
                popsize       = self.parameters.get("popsize"      , 25        )  ,
                mutation      = self.parameters.get("mutation"     , (0.5, 1.0))  ,
                recombination = self.parameters.get("recombination", 0.9       )  ,
                atol          = self.parameters.get("atol"         , 1e-4      )  ,
                disp          = self.parameters.get("disp"         , True      )  )

            self.printer.print_var("success" ,res.success          )
            self.printer.print_var("message" ,res.message          )
            self.printer.print_var("x (norm)",res.x                )
            self.printer.print_var("x (real)",self.norm2real(res.x))

            success = res.success
            x_norm  = res.x
            n_iter  = res.nit

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
