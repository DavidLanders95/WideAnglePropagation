import numpy as np
import subprocess
import os
import re

# Physical Constants (Atomic Units)
SOL_LIGHT = 137.035999
MASS_PART = 1.0
BOHR_RADIUS_ANGSTROM = 0.5291772109

def calculate_physics_params(wavelength_angstrom):
    """
    Calculate derived relativistic parameters for a given wavelength.
    """
    lambda_au = wavelength_angstrom / BOHR_RADIUS_ANGSTROM
    if lambda_au == 0:
        return 0, 0, 0

    p_inf = (2.0 * np.pi) / lambda_au

    # E_total^2 = p^2 c^2 + m^2 c^4
    term_pc = p_inf * SOL_LIGHT
    term_mc2 = MASS_PART * SOL_LIGHT**2
    E_total = np.sqrt(term_pc**2 + term_mc2**2)

    E_kinetic = E_total - term_mc2
    E_sim = 0.5 * p_inf**2

    return E_total, p_inf, E_sim

class RelativisticSolver:
    def __init__(self, source_file=None, binary_file=None):
        """Initialize solver. If `source_file` is None the packaged `fdtd.cu` is used.

        `binary_file` defaults to an executable placed next to the source file.
        """
        # Default source file lives in the package directory
        if source_file is None:
            pkg_dir = os.path.dirname(__file__)
            source_file = os.path.join(pkg_dir, "fdtd.cu")

        # Ensure absolute paths
        self.source_file = os.path.abspath(source_file)

        # Default binary next to source (named fdtd_open_boundaries)
        if binary_file is None:
            self.binary_file = os.path.join(os.path.dirname(self.source_file), "fdtd_open_boundaries")
        else:
            self.binary_file = os.path.abspath(binary_file)

        # Ensure working directory context (where potential.bin and snapshots reside)
        self.working_dir = os.path.dirname(self.source_file) or "."

    def _update_macros(self, nx, ny, xmin, xmax, ymin, ymax,
                       pml_thick=0.5, pml_sigma=80.0, pml_theta=0.785, pml_power=3.0,
                       j_source=120):
        """
        Reads C++ source, updates #define MACRO values if changed.
        Returns True if the file was modified.
        """
        if not os.path.exists(self.source_file):
            raise FileNotFoundError(f"Source file not found: {self.source_file}")

        with open(self.source_file, 'r') as f:
            content = f.read()

        new_content = content

        # Helper to safely replace defines
        def replace(name, val_str):
            # Regex searches for: #define NAME ... (until end of line)
            return re.sub(
                fr"(#define\s+{name}\s+).*?$",
                fr"\g<1>{val_str}",
                new_content,
                flags=re.MULTILINE
            )

        new_content = replace("NX", str(int(nx)))
        new_content = replace("NY", str(int(ny)))
        new_content = replace("XMIN", f"{xmin:.4f}f")
        new_content = replace("XMAX", f"{xmax:.4f}f")
        new_content = replace("YMIN", f"{ymin:.4f}f")
        new_content = replace("YMAX", f"{ymax:.4f}f")

        # PML Parameters
        new_content = replace("PML_THICK", f"{pml_thick:.4f}f")
        new_content = replace("PML_SIGMA0", f"{pml_sigma:.4f}f")
        new_content = replace("PML_THETA", f"{pml_theta:.4f}f")
        new_content = replace("PML_POWER", f"{pml_power:.1f}f")

        new_content = replace("J_SOURCE", str(int(j_source)))

        if new_content != content:
            print(f"Configuration changed. Updating {os.path.basename(self.source_file)}...")
            with open(self.source_file, 'w') as f:
                f.write(new_content)
            return True
        return False

    def compile(self):
        """Compiles the CUDA solver using nvcc."""
        print("Compiling CUDA solver...")

        # Command: nvcc -O3 -I... -o binary source
        cmd = [
            "nvcc",
            "-O3",
            "-I/usr/local/cuda/include",
            "-o", self.binary_file,
            self.source_file
        ]

        # Ensure CUDA bin is in PATH so nvcc can find cudafe++ etc
        env = os.environ.copy()
        cuda_bin = "/usr/local/cuda/bin"
        if cuda_bin not in env.get("PATH", ""):
            env["PATH"] = f"{cuda_bin}:{env.get('PATH', '')}"

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=self.working_dir, env=env)
            print(f"Compilation successful: {os.path.basename(self.binary_file)}")
        except subprocess.CalledProcessError as e:
            print("Compilation FAILED:")
            print(e.stderr)
            raise RuntimeError("Build failed")

    def run(self, V_map, wavelength_angstrom, extent,
            n_frames=600, steps_per_frame=100, snapshot_count=None,
            pml_thick=0.5, pml_sigma=80.0, j_source=120, **kwargs):
        """
        Runs the FDTD simulation.

        Parameters:
        -----------
        V_map : 2D numpy array (float32)
            The effective potential map.
        wavelength_angstrom : float
            Beam wavelength in Angstroms.
        extent : tuple (xmin, xmax, ymin, ymax)
            Physical extents of the simulation domain.
        n_frames : int
            Total number of frames to simulate.
        steps_per_frame : int, optional
            Number of time steps per frame (output).
        snapshot_count : int, optional
            Number of recent frames to keep in history (defaults to n_frames).
        pml_thick : float, optional
            Width of PML layer in a.u.
        pml_sigma : float, optional
            Strength of PML absorption.
        j_source : int, optional
            Y-index for source injection.

        Returns:
        --------
        dict : {
            "snapshots": [(t, phi, psi), ...],
            "dt": float,
            "E_sim": float,
            "success": bool
        }
        """
        ny, nx = V_map.shape
        xmin, xmax, ymin, ymax = extent

        if snapshot_count is None:
            snapshot_count = n_frames

        # Handle legacy or alternative argument names from kwargs
        if 'abs_width' in kwargs: pml_thick = kwargs['abs_width']
        if 'eta_max' in kwargs:
            # Simple conversion heuristic if user passes huge eta_max
            # If > 10000, assuming it's old style, clamp to ~100 or something reasonable?
            # Or just ignore it if it's crazy high.
            eta = kwargs['eta_max']
            if eta < 5000:
                pml_sigma = eta
            else:
                 # heuristic: 80 is good. If user says 4000000, ignore.
                 pass

        # 1. Update C++ configuration -> Recompile if needed
        macros_changed = self._update_macros(nx, ny, xmin, xmax, ymin, ymax,
                                           pml_thick=pml_thick, pml_sigma=pml_sigma,
                                           j_source=j_source)

        # Check if source code has been modified since last compile
        source_mtime = os.path.getmtime(self.source_file)
        if os.path.exists(self.binary_file):
            binary_mtime = os.path.getmtime(self.binary_file)
        else:
            binary_mtime = 0

        source_newer = source_mtime > binary_mtime

        if macros_changed or not os.path.exists(self.binary_file) or source_newer:
            self.compile()

        # 2. Write Potential to Binary
        pot_file = os.path.join(self.working_dir, "potential.bin")
        with open(pot_file, "wb") as f:
            f.write(np.array([nx, ny], dtype=np.int32).tobytes())
            f.write(V_map.astype(np.float32).tobytes())

        # 3. Run Simulation
        # Pass negative wavelength to signal Angstrom units to the C++ binary
        input_param = -abs(wavelength_angstrom)

        cmd = [
            self.binary_file,
            str(n_frames),
            str(steps_per_frame),
            str(snapshot_count),
            str(input_param)
        ]

        # Ensure executable permission
        if os.path.exists(self.binary_file):
            os.chmod(self.binary_file, 0o755)

        print(f"Running: {' '.join([os.path.basename(c) for c in cmd])}")

        try:
            # Run inside working_dir to ensure it finds potential.bin
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=self.working_dir
            )
            # print(result.stdout) # Optional: Be verbose
        except subprocess.CalledProcessError as e:
            print("Simulation Execution Failed:")
            print(e.stdout)
            print(e.stderr)
            return None

        # 4. Load Results
        snap_file = os.path.join(self.working_dir, "snapshots.bin")
        return self._load_snapshots(snap_file)

    def _load_snapshots(self, filepath):
        snapshots = []
        if not os.path.exists(filepath):
            print(f"Error: Output file {filepath} not found.")
            return None

        with open(filepath, 'rb') as f:
            # Header
            nx = np.frombuffer(f.read(4), dtype=np.int32)[0]
            ny = np.frombuffer(f.read(4), dtype=np.int32)[0]
            M = np.frombuffer(f.read(4), dtype=np.int32)[0]
            dt = np.frombuffer(f.read(4), dtype=np.float32)[0]
            E_sim = np.frombuffer(f.read(4), dtype=np.float32)[0]

            # Snapshots
            for _ in range(M):
                t = np.frombuffer(f.read(4), dtype=np.float32)[0]
                phi = np.frombuffer(f.read(nx*ny*4), dtype=np.float32).reshape((ny,nx))
                psi = np.frombuffer(f.read(nx*ny*4), dtype=np.float32).reshape((ny,nx))
                snapshots.append((t, phi, psi))

        print(f"Loaded {len(snapshots)} snapshots.")
        return {
            "snapshots": snapshots,
            "dt": dt,
            "E_sim": E_sim,
            "success": True
        }


class RelativisticSolver3D(RelativisticSolver):
    def __init__(self, source_file=None, binary_file=None):
        if source_file is None:
            pkg_dir = os.path.dirname(__file__)
            source_file = os.path.join(pkg_dir, "fdtd_3d.cu")

        super().__init__(source_file, binary_file)

        if binary_file is None:
            self.binary_file = os.path.join(os.path.dirname(self.source_file), "fdtd_3d_bin")

    def _update_macros(self, nx, ny, nz,
                       xmin, xmax, ymin, ymax, zmin, zmax,
                       pml_thick=0.5, pml_sigma=80.0, pml_theta=0.785, pml_power=3.0,
                       k_source=20,
                       wavelength_angstrom=None, use_angstrom_units=False):
        if not os.path.exists(self.source_file):
            raise FileNotFoundError(f"Source file not found: {self.source_file}")

        with open(self.source_file, 'r') as f:
            content = f.read()

        new_content = content

        def replace(name, val_str):
            return re.sub(
                fr"(#define\s+{name}\s+).*?$",
                fr"\g<1>{val_str}",
                new_content,
                flags=re.MULTILINE
            )

        # Inject explicit unit handling
        # For Angstrom/eV mode, we calculate derived constants
        if use_angstrom_units and wavelength_angstrom:
            # Physics in eV/Angstrom
            # hbar * c [eV * A]
            hc = 1973.2698
            # electron rest energy [eV]
            mc2 = 510998.95

            # Wavelength -> p
            # p * c = h * c / lambda = 2*pi * hbar * c / lambda
            # Actually p = h / lambda.   p*c = (h*c)/lambda
            # But let's look at P_INF in code. P_INF is usually wavevector k = 2pi/lambda?
            # No, in AU P_INF is momentum.
            # In the code: float phase = k_beam * (z - z0) ...
            # k_beam = sqrt(E_sim / Coeff).
            # We need k_beam = 2*pi / lambda.
            # So sqrt(E_sim / Coeff) = 2*pi/lambda
            # => E_sim = Coeff * (2*pi/lambda)^2.

            # LAP_COEFF derivation:
            # Relativistic Schrodinger Eq: (- hbar^2 c^2 / (2 E_total) * Lap + V) Psi = ...
            # So LAP_COEFF = (hbar*c)^2 / (2 * E_total) approx?
            # E_total = sqrt((pc)^2 + (mc^2)^2).
            # p = h/lambda => pc = 12398.42 / lambda

            pc = 12398.4198 / wavelength_angstrom
            E_total = np.sqrt(pc**2 + mc2**2)

            # A_coef = (hbar c)^2 / (2 * E_total)
            # hbar c = 1973.27
            A_coef = (hc**2) / (2.0 * E_total)

            p_inf_val = (2.0 * np.pi) / wavelength_angstrom

            new_content = replace("LAP_COEFF", f"{A_coef:.6f}f")
            new_content = replace("P_INF", f"{p_inf_val:.6f}f")

        else:
            # Default AU Logic (Coeff = 0.5)
            new_content = replace("LAP_COEFF", "0.5f")
            # Clear P_INF so main() calculates it if needed, or set 0
            # Ideally we remove P_INF define if it exists, or just leave as is if not used
            pass

        new_content = replace("NX", str(int(nx)))
        new_content = replace("NY", str(int(ny)))
        new_content = replace("NZ", str(int(nz)))

        new_content = replace("XMIN", f"{xmin:.4f}f")
        new_content = replace("XMAX", f"{xmax:.4f}f")
        new_content = replace("YMIN", f"{ymin:.4f}f")
        new_content = replace("YMAX", f"{ymax:.4f}f")
        new_content = replace("ZMIN", f"{zmin:.4f}f")
        new_content = replace("ZMAX", f"{zmax:.4f}f")

        new_content = replace("PML_THICK", f"{pml_thick:.4f}f")
        new_content = replace("PML_SIGMA0", f"{pml_sigma:.4f}f")
        new_content = replace("PML_THETA", f"{pml_theta:.4f}f")
        new_content = replace("PML_POWER", f"{pml_power:.1f}f")

        new_content = replace("K_SOURCE", str(int(k_source)))

        if new_content != content:
            print(f"Configuration changed. Updating {os.path.basename(self.source_file)}...")
            with open(self.source_file, 'w') as f:
                f.write(new_content)
            return True
        return False

    def run(self, V_map, wavelength_angstrom, extent,
            n_frames=600, steps_per_frame=100, snapshot_count=None,
            pml_thick=0.5, pml_sigma=80.0, k_source=20,
            use_angstrom_units=True, **kwargs):
        """
        Runs the 3D FDTD simulation.

        V_map: (NZ, NY, NX) array
               If use_angstrom_units=True, V_map should be in eV.
               If False, V_map should be in Hartrees.
        extent: (xmin, xmax, ymin, ymax, zmin, zmax)
                If use_angstrom_units=True, should be in Angstroms.
                If False, in Bohrs.
        """
        nz, ny, nx = V_map.shape
        xmin, xmax, ymin, ymax, zmin, zmax = extent

        if snapshot_count is None:
            snapshot_count = n_frames

        if 'abs_width' in kwargs: pml_thick = kwargs['abs_width']

        macros_changed = self._update_macros(nx, ny, nz,
                                             xmin, xmax, ymin, ymax, zmin, zmax,
                                             pml_thick=pml_thick, pml_sigma=pml_sigma,
                                             k_source=k_source,
                                             wavelength_angstrom=wavelength_angstrom,
                                             use_angstrom_units=use_angstrom_units)

        source_mtime = os.path.getmtime(self.source_file)
        if os.path.exists(self.binary_file):
            binary_mtime = os.path.getmtime(self.binary_file)
        else:
            binary_mtime = 0

        source_newer = source_mtime > binary_mtime

        if macros_changed or not os.path.exists(self.binary_file) or source_newer:
            self.compile()

        pot_file = os.path.join(self.working_dir, "potential.bin")
        with open(pot_file, "wb") as f:
            f.write(np.array([nx, ny, nz], dtype=np.int32).tobytes())
            f.write(V_map.astype(np.float32).tobytes())

        # If using AU, we passed -wl to indicate calc in main.
        # If using Angstrom units, we already baked physics into Macros.
        # But main() still expects an argument. We can pass anything or use it to confirm.
        input_param = -abs(wavelength_angstrom)

        cmd = [
            self.binary_file, 
            str(n_frames), 
            str(steps_per_frame), 
            str(snapshot_count), 
            str(input_param)
        ]

        if os.path.exists(self.binary_file): 
            os.chmod(self.binary_file, 0o755)

        print(f"Running 3D: {' '.join([os.path.basename(c) for c in cmd])}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=self.working_dir
            )
        except subprocess.CalledProcessError as e:
            print("Simulation Execution Failed:")
            print(e.stdout)
            print(e.stderr)
            return None

        snap_file = os.path.join(self.working_dir, "snapshots.bin")
        return self._load_snapshots(snap_file)

    def _load_snapshots(self, filepath):
        snapshots = []
        if not os.path.exists(filepath):
            print(f"Error: Output file {filepath} not found.")
            return None

        with open(filepath, 'rb') as f:
            nx = np.frombuffer(f.read(4), dtype=np.int32)[0]
            ny = np.frombuffer(f.read(4), dtype=np.int32)[0]
            nz = np.frombuffer(f.read(4), dtype=np.int32)[0]
            M = np.frombuffer(f.read(4), dtype=np.int32)[0]
            dt = np.frombuffer(f.read(4), dtype=np.float32)[0]
            E_sim = np.frombuffer(f.read(4), dtype=np.float32)[0]

            vol_size = nx*ny*nz

            for _ in range(M):
                t = np.frombuffer(f.read(4), dtype=np.float32)[0]
                phi = np.frombuffer(f.read(vol_size*4), dtype=np.float32).reshape((nz,ny,nx))
                psi = np.frombuffer(f.read(vol_size*4), dtype=np.float32).reshape((nz,ny,nx))
                snapshots.append((t, phi, psi))

        print(f"Loaded {len(snapshots)} snapshots.")
        return {
            "snapshots": snapshots,
            "dt": dt,
            "E_sim": E_sim,
            "success": True
        }
