# dolfin_warp

A set of FEniCS- and VTK-based python tools for Finite Element Digital Image Correlation/Image Registration/Motion Tracking, basically implementing the method described in [[Genet, Stoeck, von Deuster, Lee & Kozerke (2018). Equilibrated Warping: Finite Element Image Registration with Finite Strain Equilibrium Gap Regularization. Medical Image Analysis, 50, 1–22.]](https://doi.org/10.1016/j.media.2018.07.007) and [[Genet (2023). Finite strain formulation of the discrete equilibrium gap principle: application to mechanically consistent regularization for large motion tracking. Comptes Rendus Mécanique, 351, 429-458.]](https://doi.org/10.5802/crmeca.228).

The library has notably been used in:
* [[Zou, Xi, Zhao, Koh, Gao, Su, Tan, Allen, Lee, Genet & Zhong (2018). Quantification of Biventricular Strains in Heart Failure With Preserved Ejection Fraction Patient Using Hyperelastic Warping Method. Frontiers in Physiology.]](https://doi.org/10.3389/fphys.2018.01295)
* [[Finsberg, Xi, Tan, Zhong, Genet, Sundnes, Lee & Wall (2018). Efficient estimation of personalized biventricular mechanical function employing gradient-based optimization. International Journal for Numerical Methods in Biomedical Engineering.]](https://doi.org/10.1002/cnm.2982)
* [[Berberoğlu, Stoeck, Moireau, Kozerke & Genet (2019). Validation of Finite Element Image Registration‐based Cardiac Strain Estimation from Magnetic Resonance Images. PAMM.]](https://doi.org/10.1002/pamm.201900418)
* [[Finsberg, Xi, Zhao, Tan, Genet, Sundnes, Lee, Zhong & Wall (2019). Computational quantification of patient-specific changes in ventricular dynamics associated with pulmonary hypertension. American Journal of Physiology-Heart and Circulatory Physiology.]](https://doi.org/10.1152/ajpheart.00094.2019)
* [[Lee & Genet (2019). Validation of Equilibrated Warping—Image Registration with Mechanical Regularization—On 3D Ultrasound Images. Functional Imaging and Modeling of the Heart (FIMH). Cham: Springer International Publishing.]](https://doi.org/10.1007/978-3-030-21949-9_36)
* [[Škardová, Rambausek, Chabiniok & Genet (2019). Mechanical and Imaging Models-Based Image Registration. VipIMAGE 2019. Cham: Springer International Publishing.]](https://doi.org/10.1007/978-3-030-32040-9_9)
* [[Zou, Leng, Xi, Zhao, Koh, Gao, Tan, Tan, Allen, Lee, Genet & Zhong (2020). Three-dimensional biventricular strains in pulmonary arterial hypertension patients using hyperelastic warping. Computer Methods and Programs in Biomedicine.]](https://doi.org/10.1016/j.cmpb.2020.105345)
* [[Gusseva, Hussain, Friesen, Moireau, Tandon, Patte, Genet, Hasbani, Greil, Chapelle & Chabiniok (2021). Biomechanical Modeling to Inform Pulmonary Valve Replacement in Tetralogy of Fallot Patients after Complete Repair. Canadian Journal of Cardiology.]](https://doi.org/10.1016/j.cjca.2021.06.018)
* [[Berberoğlu, Stoeck, Moireau, Kozerke & Genet (2021). In-silico study of accuracy and precision of left-ventricular strain quantification from 3D tagged MRI. PLOS ONE.]](https://doi.org/10.1371/journal.pone.0258965)
* [[Castellanos, Škardová, Bhattaru, Berberoğlu, Greil, Tandon, Dillenbeck, Burkhardt, Hussain, Genet & Chabiniok (2021). Left Ventricular Torsion Obtained Using Equilibrated Warping in Patients with Repaired Tetralogy of Fallot. Pediatric Cardiology.]](https://doi.org/10.1007/s00246-021-02608-y)
* [[Berberoğlu, Stoeck, Kozerke & Genet (2022). Quantification of left ventricular strain and torsion by joint analysis of 3D tagging and cine MR images. Medical Image Analysis.]](https://doi.org/10.1016/j.media.2022.102598)
* [[Patte, Brillet, Fetita, Gille, Bernaudin, Nunes, Chapelle & Genet (2022). Estimation of regional pulmonary compliance in idiopathic pulmonary fibrosis based on personalized lung poromechanical modeling. Journal of Biomechanical Engineering.]](https://doi.org/10.1115/1.4054106)
* [[Laville, Fetita, Gille, Brillet, Nunes, Bernaudin & Genet (2023). Comparison of optimization parametrizations for regional lung compliance estimation using personalized pulmonary poromechanical modeling. Biomechanics and Modeling in Mechanobiology.]](https://doi.org/10.1007/s10237-023-01691-9)

(If you use it for your own work please let me know!)

### Tutorials

Interactive tutorials can be found at https://mgenet.gitlabpages.inria.fr/dolfin_warp-tutorials/index.html.

### Installation

A working installation of [FEniCS](https://fenicsproject.org) (version 2019.1.0; including the dolfin python interface) & [VTK](https://vtk.org) (also including python interface) is required to run `dolfin_warp`.
To setup a system, the simplest is to use [conda](https://conda.io): first install [miniconda](https://docs.conda.io/projects/miniconda/en/latest) (note that for Microsoft Windows machines you first need to install WSL, the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install), and then install miniconda for linux inside the WSL; for Apple MacOS machines with Apple Silicon CPUs, you still need to install the MacOS Intel x86_64 version of miniconda), and then install the necessary packages:
```
conda create -y -c conda-forge -n dolfin_warp fenics=2019.1.0 gnuplot=5.4 meshio=5.3 mpi4py=3.1.3 pip python=3.10 vtk=9.1
conda activate dolfin_warp
conda env config vars set CPATH=$CONDA_PREFIX/include/vtk-9.1
conda activate dolfin_warp
pip install dolfin_warp
```
