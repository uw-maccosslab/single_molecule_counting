# read version from installed package
from importlib.metadata import version
__version__ = version("single_mol")

# import functions
from single_mol.single_mol import fig2a, fig2a_insert
from single_mol.single_mol import fig2b, fig2c, fig2d
