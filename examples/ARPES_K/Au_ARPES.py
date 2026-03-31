from ase2sprkkr import SPRKKR
ARPES_input={
'CONTROL.PRINT': 0,
'CONTROL.DATASET':'Au',
'TAU.NKTAB':500,
'SITES.NL':4,
'ENERGY.ImE':0.0,
'ENERGY.EMINEV':0.0,
# Minimum energy
'ENERGY.EMAXEV':0.0,
# Maximum energy
'ENERGY.EWORK_EV':5.0,
# Work function
'ENERGY.NE':1,
# Number of energy points
'ENERGY.IMV_INI_EV':0.02,
# Imaginary part of initial state
'ENERGY.IMV_FIN_EV':2.0,
# Imaginary part of final state
'ENERGY.GRID':1,
'TASK.STRVER':1,
'TASK.IQ_AT_SURF':1,
# Site number at surface layer
'TASK.MILLER_HKL':[1,1,1], # Miller indices of surface
'SPEC_PH.THETA':45,
# Polar angle of photon beam
'SPEC_PH.PHI':90,
# Azimuthal angle of photon beam
'SPEC_PH.EPHOT':50,
# Photon energy
'SPEC_STR.N_LAYDBL':[12,12],
'SPEC_STR.NLAT_G_VEC':37,
'SPEC_STR.N_LAYER':30,
'SPEC_STR.SURF_BAR':[0.275,0.275],
'SPEC_EL.KA':[-0.5,-0.5],
# Corner of k-grid
'SPEC_EL.K1':[1,0],
# First vector of k-grid
'SPEC_EL.K2':[0,1],
# Second vector of k-grid
'SPEC_EL.NK1':120,
# Number k-points along K1
'SPEC_EL.NK2':120,
# Number k-points along K2
'SPEC_EL.SPOL':4,
'SPEC_EL.PSPIN':[0,0,1]
}
calculator=SPRKKR(potential='Au.pot_new',
input_parameters='arpes',
options=ARPES_input)
calculator.calculate()
