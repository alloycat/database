mport numpy as np
from ase.build import fcc100
from ase.constraints import FixAtoms
from ase.io import write
from ase.build import sort

seed=441331223 
host='Au'      #metal 1#
impurity='Pd'  #metal 2#

latticeconstant=3.8907*0.25+4.0782*0.75  #Pd=3.8907; Au=4.0782, if Pd%=0.25, then is 3.8907*0.25+4.0782*0.75 #  Rh= 3.8034  ;  Pt= 3.9242 ; Cu= 3.6149 ; Ag= 4.0853
concentration_impurity=0.25  #the composition of metal 2#

model=fcc100(host, size=(4,4,4), a=latticeconstant, vacuum=6.0, orthogonal=True)
c = FixAtoms(mask=[x >2   for x in model.get_tags()])
model.set_constraint(c)

elements=model.get_chemical_symbols()
num_atom=model.get_number_of_atoms()

num_impurity=np.round(num_atom*concentration_impurity)


np.random.seed(seed)

i=0
while i < int(num_impurity):
    r=np.random.rand()
    n=int(np.round(r*num_atom))
    if elements[n]==host:
        elements[n]=impurity
        i=i+1


model.set_chemical_symbols(elements)
model=sort(model)
write('POSCAR',model,format='vasp',direct=True)
