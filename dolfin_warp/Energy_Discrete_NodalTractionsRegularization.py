#coding=utf8

################################################################################
###                                                                          ###
### dolfin_warp                                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy
import petsc4py
import typing

import dolfin_mech as dmech

from .Energy               import Energy
from .EnergyMixin_Discrete import DiscreteEnergyMixin
from .Problem              import Problem

################################################################################

class NodalTractionsRegularizationDiscreteEnergy(Energy, DiscreteEnergyMixin):
    """Boundary tractions regularization based on the discrete nodal forces.

    The discrete tractions are defined at the boundary nodes from the internal
    nodal forces, i.e., the boundary rows of the discrete equilibrium residual
    f = int P : grad(phi) dV (which the volume equilibrium gap term discards),
    divided by the lumped nodal boundary area. Unlike the tractions computed as
    P.N on exterior facets, they involve every element, including the elements
    whose vertices all lie on the boundary but that have no exterior facet
    (e.g., one third of the elements of a single layer of prisms split into
    tetrahedra), which otherwise contribute to no discrete term.

    As for the facet-based terms, the surface gradient of the normal component
    ("tractions-normal") or of the tangential vector ("tractions-tangential",
    with covariant projection, i.e., without the curvature term) of these nodal
    tractions is penalized. The splitting uses nodal reference normals.

    Since f is linear in the nodal forces, R = L.f(U) with L a constant matrix,
    so that the energy is 1/2 R^T M^-1 R, the residual is K^T L^T M^-1 R, and
    the (Gauss-Newton) jacobian is K^T H K, with K = df/dU and H = L^T M^-1 L
    assembled once."""



    def __init__(self,
            problem                  : Problem                                                          ,
            name                     : str                            = "reg"                           ,
            w                        : float                          = 1.                              ,
            type                     : str                            = "tractions-normal"              ,
            model                    : str                            = "ogdenciarletgeymonatneohookean",
            young                    : float                          = 1.                              ,
            poisson                  : float                          = 0.                              ,
            b_fin                    : typing.Optional["list[float]"] = None                            ,
            quadrature_degree        : typing.Optional[int]           = None                            ,
            volume_subdomain_data                                     = None                            ,
            volume_subdomain_id                                       = None                            ,
            surface_subdomain_data                                    = None                            ,
            surface_subdomain_id                                      = None                            ):

        self.problem = problem
        self.printer = problem.printer

        self.name = name

        self.w = w

        type_lst = ("tractions-normal", "tractions-tangential")
        assert (type in type_lst),\
            "\"type\" ("+str(type)+") must be in "+str(type_lst)+". Aborting."
        self.type = type

        self.model = model
        self.young = young
        self.poisson = poisson

        self.printer.print_str("Defining regularization energy…")
        self.printer.inc()

        mesh = self.problem.mesh
        self.dim = self.problem.mesh_dimension

        self.dV = dolfin.Measure(
            "dx",
            domain=mesh,
            subdomain_data=volume_subdomain_data,
            subdomain_id=volume_subdomain_id if volume_subdomain_id is not None else "everywhere",
            metadata={"quadrature_degree":quadrature_degree})
        self.dS = dolfin.Measure(
            "ds",
            domain=mesh,
            subdomain_data=surface_subdomain_data,
            subdomain_id=surface_subdomain_id if surface_subdomain_id is not None else "everywhere",
            metadata={"quadrature_degree":quadrature_degree})

        # nodal internal forces
        if (self.model == "hooke"):
            self.kinematics = dmech.LinearizedKinematics(
                u=self.problem.U)
            dE_test = dolfin.derivative(
                self.kinematics.epsilon, self.problem.U, self.problem.dU_test)
        else:
            self.kinematics = dmech.Kinematics(
                U=self.problem.U)
            dE_test = dolfin.derivative(
                self.kinematics.E, self.problem.U, self.problem.dU_test)
        self.material = dmech.material_factory(
            kinematics=self.kinematics,
            model=self.model,
            parameters={
                "E":self.young,
                "nu":self.poisson,
                "checkJ":1})
        self.Wint_form = dolfin.inner(self.material.Sigma, dE_test) * self.dV
        self.dWint_form = dolfin.derivative(self.Wint_form, self.problem.U, self.problem.dU_trial)
        self.f_vec = self.problem.U.vector().copy()
        self.K_mat = dolfin.PETScMatrix()

        # nodal boundary areas & normals
        n_vertices = mesh.num_vertices()
        vertex_ds = dolfin.ds(
            domain=mesh,
            subdomain_data=surface_subdomain_data,
            subdomain_id=surface_subdomain_id if surface_subdomain_id is not None else "everywhere",
            scheme="vertex",
            metadata={
                "degree":1,
                "representation":"quadrature"})
        Vs = dolfin.FunctionSpace(mesh, "Lagrange", 1)
        Vv = dolfin.VectorFunctionSpace(mesh, "Lagrange", 1)
        VT = dolfin.TensorFunctionSpace(mesh, "Lagrange", 1)
        A = dolfin.assemble(dolfin.TestFunction(Vs) * vertex_ds).get_local()[dolfin.vertex_to_dof_map(Vs)]
        N = dolfin.assemble(dolfin.inner(self.problem.N, dolfin.TestFunction(Vv)) * vertex_ds).get_local()[dolfin.vertex_to_dof_map(Vv)].reshape(n_vertices, self.dim)
        N_norm = numpy.linalg.norm(N, axis=1)
        is_boundary = (A > 0.) & (N_norm > 0.)
        N[is_boundary] /= N_norm[is_boundary,None]
        P = numpy.eye(self.dim)[None,:,:] - numpy.einsum("vi,vj->vij", N, N) # nodal tangent plane projectors (identity at interior nodes)

        U_v2d  = dolfin.vertex_to_dof_map(self.problem.U_fs).reshape(n_vertices, self.dim)
        Vs_v2d = dolfin.vertex_to_dof_map(Vs)
        Vv_v2d = dolfin.vertex_to_dof_map(Vv).reshape(n_vertices, self.dim)
        VT_v2d = dolfin.vertex_to_dof_map(VT).reshape(n_vertices, self.dim, self.dim)

        def new_mat(n_rows, n_cols, nnz):
            mat = petsc4py.PETSc.Mat().createAIJ([n_rows, n_cols], nnz=nnz)
            mat.setUp()
            mat.setOption(petsc4py.PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            return mat

        n_U = self.problem.U.vector().size()
        proj_op = dolfin.Identity(self.dim) - dolfin.outer(self.problem.N, self.problem.N)
        if (self.type == "tractions-normal"):
            # S: nodal forces -> nodal normal tractions (scalar P1)
            S = new_mat(Vs.dim(), n_U, self.dim)
            for v in numpy.where(is_boundary)[0]:
                S.setValues([int(Vs_v2d[v])], [int(d) for d in U_v2d[v]], N[v]/A[v])
            S.assemble()
            # B: nodal normal tractions -> weak surface gradient (vector P1)
            t_trial = dolfin.TrialFunction(Vs)
            v_test = dolfin.TestFunction(Vv)
            divs_v_test = dolfin.tr(dolfin.dot(proj_op, dolfin.dot(dolfin.grad(v_test), proj_op)))
            B = dolfin.as_backend_type(dolfin.assemble(dolfin.inner(t_trial, divs_v_test) * self.dS)).mat()
            # Q: projection of the gradient onto the nodal tangent plane: the weak
            # surface gradient of a constant field is not zero on a curved surface,
            # int grad_s(phi) dA = int phi kappa N dA (kappa mean curvature), so
            # that a uniform pressure would otherwise be penalized.
            Q = new_mat(Vv.dim(), Vv.dim(), self.dim)
            for v in range(n_vertices):
                Pv = P[v] if is_boundary[v] else numpy.eye(self.dim)
                for j in range(self.dim):
                    Q.setValues([int(Vv_v2d[v,j])], [int(d) for d in Vv_v2d[v]], Pv[j,:])
            Q.assemble()
            L = Q.matMult(B.matMult(S))
            M_diag = numpy.zeros(Vv.dim())
            for v in range(n_vertices):
                M_diag[Vv_v2d[v]] = A[v]
        elif (self.type == "tractions-tangential"):
            # S: nodal forces -> nodal tangential tractions (vector P1)
            S = new_mat(Vv.dim(), n_U, self.dim)
            for v in numpy.where(is_boundary)[0]:
                for i in range(self.dim):
                    S.setValues([int(Vv_v2d[v,i])], [int(d) for d in U_v2d[v]], P[v,i,:]/A[v])
            S.assemble()
            # B: nodal tangential tractions -> weak surface gradient (tensor P1)
            t_trial = dolfin.TrialFunction(Vv)
            T_test = dolfin.TestFunction(VT)
            divs_T_test = dolfin.as_vector([dolfin.tr(dolfin.dot(proj_op, dolfin.dot(dolfin.grad(T_test[i,:]), proj_op))) for i in range(self.dim)])
            B = dolfin.as_backend_type(dolfin.assemble(dolfin.inner(t_trial, divs_T_test) * self.dS)).mat()
            # Q: covariant projection, of the traction component index (removes the
            # -N⊗(B.t) curvature term) and of the gradient index (removes the
            # mean curvature term of the weak surface gradient), onto the nodal
            # tangent plane
            Q = new_mat(VT.dim(), VT.dim(), self.dim**2)
            for v in range(n_vertices):
                Pv = P[v] if is_boundary[v] else numpy.eye(self.dim)
                PP = numpy.einsum("ik,jl->ijkl", Pv, Pv)
                cols = [int(VT_v2d[v,k,l]) for k in range(self.dim) for l in range(self.dim)]
                for i in range(self.dim):
                    for j in range(self.dim):
                        Q.setValues([int(VT_v2d[v,i,j])], cols, PP[i,j].flatten())
            Q.assemble()
            L = Q.matMult(B.matMult(S))
            M_diag = numpy.zeros(VT.dim())
            for v in range(n_vertices):
                M_diag[VT_v2d[v].flatten()] = A[v]
        M_inv_diag = numpy.zeros_like(M_diag)
        M_inv_diag[M_diag > 0.] = 1./M_diag[M_diag > 0.]
        M_inv = petsc4py.PETSc.Vec().createWithArray(M_inv_diag)

        self.L = L
        self.M_inv = M_inv
        self.R = L.createVecLeft()
        self.MR = L.createVecLeft()
        LT_Minv = L.copy().transpose()
        LT_Minv.diagonalScale(R=M_inv)
        self.H = LT_Minv.matMult(L) # L^T M^-1 L, constant
        self.g = L.createVecRight()

        self.res_vec = self.problem.U.vector().copy()

        self.printer.dec()



    def update_R(self):

        dolfin.assemble(
            form=self.Wint_form,
            tensor=self.f_vec)
        self.L.mult(self.f_vec.vec(), self.R)
        self.MR.pointwiseMult(self.M_inv, self.R)



    def update_ener(self):

        self.update_R()
        ener = self.R.dot(self.MR)/2
        return ener



    def update_res(self):

        self.update_R()
        self.L.multTranspose(self.MR, self.g) # g = L^T M^-1 R
        dolfin.assemble(
            form=self.dWint_form,
            tensor=self.K_mat)
        self.K_mat.mat().multTranspose(self.g, self.res_vec.vec())



    def update_jac(self):

        dolfin.assemble(
            form=self.dWint_form,
            tensor=self.K_mat)
        if not hasattr(self, "jac_mat"):
            self.jac_mat = dolfin.PETScMatrix(self.H.PtAP(self.K_mat.mat()))
        else:
            self.H.PtAP(self.K_mat.mat(), result=self.jac_mat.mat())
