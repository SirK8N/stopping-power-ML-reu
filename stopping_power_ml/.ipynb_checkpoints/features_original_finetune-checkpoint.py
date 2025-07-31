"""Functions related to computing features"""
import abc
import itertools
from scipy.integrate import romb
import numpy as np
from matminer.featurizers.site import AGNIFingerprints
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
import copy
from stopping_power_ml.rc import *

def insert_projectile(atoms, projectile_species, position):
    """Add the projectile at a certain position into the primitive cell
    :param projectile_species: str
    :param position: [float]*3, projectile position in cartesian coordinates
    :return: Structure, output of the cell"""

    atoms.append(projectile_species, position, coords_are_cartesian = True)
    return atoms

def split_atoms_based_on_species(atoms):
    """
    split the ase atoms based on the species
    """
    original_cell = atoms.get_cell()
    original_pbc = atoms.get_pbc()
    unique_symbols = np.unique(atoms.get_chemical_symbols())

    split_atoms = []

    for symbol in unique_symbols:
        species_indices = [atom.index for atom in atoms if atom.symbol == symbol]
        species_atoms = atoms[species_indices]

        split_atoms.append(Atoms(symbols=species_atoms.get_chemical_symbols(),
                                      positions=species_atoms.get_positions(),
                                      cell=original_cell,
                                      pbc=original_pbc))
    return split_atoms;


class ProjectileFeaturizer(BaseFeaturizer):
    """Abstract base class for computing features about a particle traveling in a material.

    Handles determining the primitive cell of a material, adding projectile to the simulation cell, etc."""

    def __init__(self, simulation_cell, use_prim_cell=True):
        """
        :param simulation_cell: ase.Atoms, simulation cell
        :param use_prim_cell: bool, whether to use primitive cell in calculation
        """

        # Compute the primitive unit cell vectors (structure minus the projectile.
        self.simulation_cell = AseAtomsAdaptor.get_structure(simulation_cell)

        self.use_prim_cell = use_prim_cell
        if use_prim_cell:
            # We use the `get_primitive_structure()` operation because it does not
            #  translate the atoms (spglib will). Translations mean the cartesian coordinates
            #  in the simulation cell and primitive cell are not the same, which causes all
            #  kinds of problems
            self.prim_cell = self.simulation_cell.get_primitive_structure()

    def _insert_projectile(self, position):

        x = self.prim_cell.copy() if self.use_prim_cell else self.simulation_cell.copy()
        return insert_projectile(x, 'H', position)

    @abc.abstractmethod
    def featurize(self, position, velocity):
        """Compute features for a projectile system"""

        raise NotImplementedError()
    
    def implementors(self):
        return ['Logan Ward']

    def citations(self):
        return []


class IonIonForce(ProjectileFeaturizer):
    """Compute the stopping force acting on a particle from ion-ion repulsion
    
    Computes the force from the repulsion of nuclei (i.e., the charge on each atom is 
    its atomic number) projected along the particle's direction of travel. 
    
    Input: Position and velocity of projectile
    
    Parameters:
        acc - float, accuracy of the Ewald summation (default=3)"""

    def __init__(self, simulation_cell, acc=3, **kwargs):
        super(IonIonForce, self).__init__(simulation_cell, **kwargs)
        self.acc = acc

    def feature_labels(self):
        return ["ion-ion_x", "ion-ion_y", "ion-ion_z"]

    def featurize(self, position, velocity):
        # Get the atoms object as a pymatgen Structure
        strc = self._insert_projectile(position)

        # Convert lattice from Bohr to Angstrom
        strc.scale_lattice((0.529177 ** 3) * strc.volume)

        # Assign a charge of Z to each atom
        for site in strc.sites:
            site.charge = site.specie.Z

        # Compute the forces
        ewald = EwaldSummation(strc, compute_forces=True, acc_factor=self.acc)

        # Compute force
        my_force = ewald.forces[-1, :] * 0.01944688972 #from eV/Angsrtom to Hartree/Bohr
        return list(my_force)

    def implementors(self):
        return ['Logan Ward']

    def citations(self):
        return []


class LocalChargeDensity(ProjectileFeaturizer):
    """
Compute the local electronic charge density around a projectile.
    """

    def __init__(self, simulation_cell, charge, **kwargs):
        super(LocalChargeDensity, self).__init__(simulation_cell, **kwargs)
        self.charge = charge

    def feature_labels(self):
        return ['charge density']

    def featurize(self, position, velocity):
        # Convert to reduced coordinates
        cur_pos = self.simulation_cell.lattice.get_fractional_coords(position) % 1
        return np.log([self.charge(cur_pos)])

    def implementors(self):
        return ['Logan Ward']

    def citations(self):
        return []

class ProjectedAGNIFingerprints(ProjectileFeaturizer):
    """Compute the fingerprints of the local atomic environment using the AGNI method


    Input: Position and velocity of projectile

    Parameters:
        etas - list of floats, window sizes used in fingerprints
        cutoff - float, cutoff distance for features
    """

    def __init__(self, simulation_cell, etas, cutoff=16, **kwargs):
        super(ProjectedAGNIFingerprints, self).__init__(simulation_cell, **kwargs)
        self.agni = AGNIFingerprints(directions=['x','y','z', None], etas = etas, cutoff = cutoff)
        self.atoms_list = [AseAtomsAdaptor.get_structure(atom) for atom in split_atoms_based_on_species(simulation_cell)]
        assert len(self.atoms_list) > 0, "has to have at least one atom in the cell"

        for atom in self.atoms_list: 
            logging.info("splitted atom")
            print(atom.lattice)
            for i, site in enumerate(atom.sites):
                print(f"{site.specie} {atom.cart_coords[i][0]:0.8f} {atom.cart_coords[i][1]:0.8f} {atom.cart_coords[i][2]:0.8f}")

    @property
    def etas(self):
        return self.agni.etas

    @etas.setter
    def etas(self, x):
        self.agni.etas = x

    @property
    def cutoff(self):
        return self.agni.cutoff

    @cutoff.setter
    def cutoff(self, x):
        self.agni.cutoff = x

    def feature_labels(self):
        labels = []
        for atom in self.atoms_list:
            symbol = atom.sites[0].specie.symbol 
            for eta in self.agni.etas:
                for d in ['x', 'y', 'z']:
                    labels.append(f'AGNI_{d} in {symbol} eta={eta:.2e}')
        return labels

    def featurize(self, position, velocity):
        proj_fingerprints = []
        for atom in self.atoms_list:
            strc = insert_projectile(atom.copy(), 'H', position)
            fingerprints = self.agni.featurize(strc, -1).reshape((4, -1)).T
            proj_fingerprints.extend(fingerprints[:, :-1].flatten())
        return proj_fingerprints

    def implementors(self):
        return ['Logan Ward']

    def citations(self):
        return []


class RepulsionFeatures(ProjectileFeaturizer):
    """Compute features the $1/r^n$ repulsion. Designed to be a faster approximation of the Coulomb repulsion force

    Input: Position and velocity of projectile

    Parameters:
        cutoff - float, cutoff distance for potential
        n - int, exponent for the repulsion potential"""

    def __init__(self, simulation_cell, cutoff=40, n=6):
        super(RepulsionFeatures, self).__init__(simulation_cell)
        self.cutoff = cutoff
        self.n = n

    def feature_labels(self):
        return ["repulsion force", ]

    def featurize(self, position, velocity):
        # Putting these temporarily in here
        strc = self._insert_projectile(position)
        proj = strc[-1]

        # Compute the 'force' acting on the projectile
        force = np.zeros(3)
        for n, r in strc.get_neighbors(proj, self.cutoff):
            disp = n.coords - proj.coords
            force += disp * proj.specie.Z * n.specie.Z / np.power(r, self.n + 1)
        return force

    def implementors(self):
        return ['Logan Ward', ]

    def citations(self):
        return []


class ProjectileVelocity(ProjectileFeaturizer):
    """Compute the projectile velocity
    
    Input: velocity of projectile. possibly take the direction of velocity if its magnitude is 0
    
    Parameters: None"""

    def feature_labels(self):
        return ["v_mag"]

    def featurize(self, position, velocity, vdir = None):
        
        v_mag = velocity
        return [v_mag]
    
def make_cartesian_grid_offsets(radius, resolution):
    '''
    find cartesian grid for offsets
    '''

    values = np.arange(-radius, radius + resolution, resolution)
    return [
        np.array([x, y, z])
        for x, y, z in itertools.product(values, values, values)
        if not (x == 0 and y == 0 and z == 0)
    ]

def make_spherical_shell_offsets(radius, n_points):
    '''
    find offsets in spherical coordinates
    '''
    
    offsets = []
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    for i in range(n_points):
        theta = 2 * np.pi * i / phi
        z = 1 - 2 * i / (n_points - 1)
        r_xy = np.sqrt(1 - z**2)
        x = r_xy * np.cos(theta)
        y = r_xy * np.sin(theta)
        offsets.append(np.array([x, y, z]) * radius)
    
    return offsets

def make_spherical_shell_band_offsets(r_min, r_max, n_points):
    """
    Generate uniformly distributed points within a spherical shell band (volume between r_min and r_max).
    
    Args:
        r_min (float): inner radius of shell
        r_max (float): outer radius of shell
        n_points (int): number of offset points to generate

    Returns:
        list of np.array([dx, dy, dz]) offset vectors
    """
    offsets = []
    for _ in range(n_points):
        # Uniform spherical direction
        phi = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.random.uniform(-1, 1)
        theta = np.arccos(cos_theta)

        # Uniform in volume: r^3 distribution
        u = np.random.uniform(r_min**3, r_max**3)
        r = u ** (1/3)

        # Convert to Cartesian
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        offsets.append(np.array([x, y, z]))

    return offsets

class PositionOffset(ProjectileFeaturizer):
    
    def __init__(self, structure, featurizer, offsets=None):
        super().__init__(structure)
        self.structure = structure
        self.featurizer = featurizer
        
        shell_offsets = offsets = make_spherical_shell_offsets(radius=0.1, n_points=8)
        self.offsets = [np.array([0.0, 0.0, 0.0])] + shell_offsets

        
    def featurize(self, position, velocity=None):
        return np.ravel([
            self.featurizer.featurize(position + offset, velocity)
            for offset in self.offsets
        ])
    def feature_labels(self):
        base_labels = self.featurizer.feature_labels()
        labels = []
        
        for offset in self.offsets:
            offset_str = f"({offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f})"
            for label in base_labels:
                labels.append(f"{label} at offset {offset_str}")
        return labels
         
    def implementors(self):
        return []

    def citations(self):
        return []
    
class PoisitonAverage(ProjectileFeaturizer):
    def __init__(self, structure, featurizer, offsets, decay=2.0):
        super().__init__(structure)
        self.featurizer = featurizer
        self.offsets = offsets
        self.decay = decay
        
    def featurize(self, position, velocity):
        features = []
        weights = []

        for offset in self.offsets:
            distance = np.linalg.norm(offset)
            weight = np.exp(-self.decay * distance)

            feature = self.featurizer.featurize(position + offset, velocity)
            weighted_feature = np.multiply(feature, weight)

            features.append(weighted_feature)
            weights.append(weight)

        # Weighted average: sum(weighted_features) / sum(weights)
        features = np.array(features)
        weights = np.array(weights)[:, np.newaxis]
        avg_features = np.sum(features, axis=0) / np.sum(weights)

        return avg_features.tolist()

    def feature_labels(self):
        return [f"position-avg {label}" for label in self.featurizer.feature_labels()]
        
    
class TimeOffset(ProjectileFeaturizer):
    """Compute the value of a feature at a different time
    
    The environment of the projectile is determined by using the 
    known velocity of the projectile."""
    
    def __init__(self, structure, featurizer, offsets=(-4,-3,-2,-1,-0.5,0,0.5,1,2)):
        """Initailize the featurizer
        
        Args:
            structure (Structure) - Structure to featurizer
            featurizer (ProjectileFeaturizer) - Featurizer to use
            offsets ([float]) - Times relative to present at which to compute features
            """
        self.structure = structure
        self.featurizer = featurizer
        self.offsets = offsets
        
    def featurize(self, position, velocity):
        positions = np.array(self.offsets)[:, np.newaxis] * \
            np.array([velocity] * len(self.offsets)) + position
        return np.ravel([self.featurizer.featurize(p, velocity) for p in positions])
    
    def feature_labels(self):
        return ['{} at t={:.2f}'.format(f, t) for t, f in itertools.product(self.offsets,                                       self.featurizer.feature_labels())]

    def implementors(self):
        return ['Logan Ward']

    def citations(self):
        return []
    

class TimeAverage(ProjectileFeaturizer):
    """Compute a weighted average of a feature over time

    The weight of events are weighted by an expontial of their time from
    the present. Users can set weights that determine whether the average of
    features in the future are past are taken into account, how how quickly
    the weights change."""

    def __init__(self, structure, featurizer, strengths=(1, 2, 3, 4, -1, -2),
                 k=5):
        """Initialize the featurizer

        Argss:
            structure (Structure) - Structure to featurizer
            featurizer (ProjectileFeaturizer) - Featurizer to average
            strengths ([float]) - How strongly features contributions varies
                with time from present. Positive weights mean the average
                will be over past events, positive ones deal with the future
            k (float) - 2 ** k + 1 points will be used in average"""
        super(TimeAverage, self).__init__(structure, True)
        self.featurizer = featurizer
        self.strengths = strengths
        self.k = k

    def featurize(self, position, velocity):

        outputs = []
        for s in self.strengths:
            # Determine particle positions and weights
            times = np.linspace(-10/s, 0, 2 ** self.k + 1)
            cur_pos = times[:, np.newaxis] * np.array([velocity] * len(times)) \
                      + position
            dt = abs(times[1] - times[0])
            weights = np.exp(times * s)

            # Evaluate the features at each of these times
            #  Do not use featurize_many because it is parallel
            #  Multiply features by the weights to prepare for integration
            features = [np.multiply(self.featurizer.featurize(pos, velocity),
                        w) for pos, w in zip(cur_pos, weights)]

            # Determine the average using Romberg integration
            outputs.append(romb(features, dx=dt, axis=0))

        # Flatten the output
        return np.squeeze(np.hstack(outputs)).tolist()

    def feature_labels(self):
        return ['time average of {}, strength={:.2f}'.format(f, s)
                for s, f in itertools.product(self.strengths,
                                              self.featurizer.feature_labels())]
