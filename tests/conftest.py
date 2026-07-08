"""Shared fixtures for wide-angle propagation tests."""
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def beam_amplitude_normalized(psi_xy, h, k, use_fftshift=True):
    """Extract normalized Fourier beam amplitude |C_{h,k}| from a real-space wave."""
    Ny, Nx = psi_xy.shape
    C = np.fft.fft2(psi_xy) / (Nx * Ny)
    if use_fftshift:
        C = np.fft.fftshift(C)
        cy, cx = Ny // 2, Nx // 2
        return np.abs(C[cy + k, cx + h])
    else:
        return np.abs(C[k % Ny, h % Nx])


# ---------------------------------------------------------------------------
# Paper reference data (extracted from Rother & Scheerschmidt 2009,
# doi:10.1016/j.ultramic.2008.08.008, Figure 3)
# ---------------------------------------------------------------------------

_raw_data_Au_Beam_0_0_Klein_Gordon_MS = """0.06421232876712413, 0.9984732824427482
0.98458904109589, 0.9874045801526719
1.9049657534246576, 0.9599236641221375
2.397260273972603, 0.9431297709923665
3.4460616438356153, 0.9041984732824428
4.216609589041096, 0.8809160305343512
4.880136986301369, 0.867557251908397
5.736301369863013, 0.8641221374045802
6.6138698630137, 0.8725190839694656
7.4486301369863, 0.8858778625954199
8.197773972602743, 0.9022900763358779
8.94691780821918, 0.918320610687023
9.76027397260274, 0.9286259541984734
10.637842465753426, 0.9324427480916031
11.665239726027401, 0.9240458015267177
12.671232876712331, 0.9041984732824428
13.613013698630137, 0.8809160305343513
14.511986301369863, 0.8595419847328245
15.325342465753433, 0.8450381679389314
16.438356164383563, 0.835496183206107
17.529965753424662, 0.8381679389312977
18.728595890410965, 0.8461832061068704
19.58476027397261, 0.8515267175572521
20.71917808219178, 0.8530534351145038
21.72517123287672, 0.8473282442748091
23.073630136986306, 0.8297709923664123
24.14383561643836, 0.8118320610687024
24.807363013698637, 0.8030534351145039"""

_raw_data_Au_Beam_0_28_Klein_Gordon_MS = """0.06620984763060989, 0.00009836065573769898
0.8162989948053649, 0.0008196721311475447
1.478421564942515, 0.0018360655737704873
2.9133055772400027, 0.0044262295081967246
5.253876697410391, 0.009475409836065572
6.732466919170015, 0.011540983606557375
8.12178949701719, 0.012131147540983607
9.796937192201312, 0.011672131147540982
11.870355914071766, 0.013180327868852457
13.613857807846884, 0.015737704918032787
14.606836865488965, 0.016983606557377046
15.841790268019778, 0.017508196721311473
16.76733069264946, 0.016983606557377046
18.177205308352853, 0.015540983606557375
19.939813610123274, 0.014098360655737704
22.034724029259564, 0.01485245901639344
23.071144264222596, 0.015213114754098356
24.085157236341217, 0.015081967213114753
24.92234558263702, 0.014327868852459017"""


def _parse_and_interpolate(raw_data):
    from scipy.interpolate import interp1d
    data = np.array(
        [[float(v) for v in line.split(",")] for line in raw_data.strip().split("\n")]
    )
    data = data[data[:, 0].argsort()]
    return interp1d(data[:, 0], data[:, 1], kind="linear", fill_value="extrapolate")


@pytest.fixture(scope="session")
def paper_beam_0_0_kg_ms():
    return _parse_and_interpolate(_raw_data_Au_Beam_0_0_Klein_Gordon_MS)


@pytest.fixture(scope="session")
def paper_beam_0_28_kg_ms():
    return _parse_and_interpolate(_raw_data_Au_Beam_0_28_Klein_Gordon_MS)


# ---------------------------------------------------------------------------
# Crystal & potential fixtures
# ---------------------------------------------------------------------------

AU_LATTICE_PARAM = 4.076
AU_ENERGY = 300e3
AU_GPTS = (128, 128)
AU_N_SLICES_PER_CELL = 2


@pytest.fixture(scope="session")
def au_atoms():
    from ase.build import bulk
    atoms = bulk("Au", "fcc", a=AU_LATTICE_PARAM, cubic=True)
    atoms.info["thermal_sigma"] = 0.0
    atoms.arrays["thermal_sigma"] = np.zeros(len(atoms))
    return atoms


def _make_abtem_potential(atoms, parametrization="lobato"):
    """Build abTEM potential (requires GPU / cupy)."""
    import abtem
    abtem.config.set({"device": "gpu"})
    abtem.config.set({"precision": "float64"})

    slice_dz = float(atoms.cell[2, 2]) / AU_N_SLICES_PER_CELL

    pot = abtem.Potential(
        atoms,
        gpts=AU_GPTS,
        slice_thickness=slice_dz,
        projection="infinite",
        parametrization=parametrization,
    )
    return pot, slice_dz


@pytest.fixture(scope="session")
def au_potential_lobato(au_atoms):
    """Potential array (V/Å) and slice thickness from Lobato parametrization."""
    jnp = pytest.importorskip("jax.numpy")
    cupy = pytest.importorskip("cupy")
    pot_obj, slice_dz = _make_abtem_potential(au_atoms, "lobato")
    pot_array = jnp.array(cupy.asnumpy(pot_obj.build(lazy=False).array / slice_dz))
    return pot_array, slice_dz


@pytest.fixture(scope="session")
def au_potential_wk(au_atoms):
    """Potential array (V/Å) and slice thickness from Weickenmeier-Kohl parametrization."""
    jnp = pytest.importorskip("jax.numpy")
    cupy = pytest.importorskip("cupy")
    from tests.wk_parametrization import make_wk_parametrization
    wk_param = make_wk_parametrization()
    pot_obj, slice_dz = _make_abtem_potential(au_atoms, wk_param)
    pot_array = jnp.array(cupy.asnumpy(pot_obj.build(lazy=False).array / slice_dz))
    return pot_array, slice_dz


@pytest.fixture(scope="session")
def au_sampling(au_atoms):
    """Pixel sizes (dy, dx) for the Au crystal on the 128x128 grid."""
    cell = au_atoms.get_cell()
    dy = float(cell[0, 0]) / AU_GPTS[0]
    dx = float(cell[1, 1]) / AU_GPTS[1]
    return (dy, dx)


@pytest.fixture(scope="session")
def plane_wave_128():
    """Uniform plane wave on 128x128 grid (complex128)."""
    jnp = pytest.importorskip("jax.numpy")
    return jnp.ones(AU_GPTS, dtype=jnp.complex128)
