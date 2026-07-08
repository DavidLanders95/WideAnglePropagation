# Reconstruction du notebook Axel/Lubk après déconnexion serveur

Date : 2026-06-26  
Mots-clés : `01_axel_lubk_verification.ipynb`, Axel Lubk, sampling, ODE, 128, 414, AS, MS, WPM

Notebook concerné :

```text
C:\Users\jr122738\Documents\New project\WideAnglePropagation\notebooks\verification\01_axel_lubk_verification.ipynb
```

Objectif pratique :

- refaire un run `128_with_ode` avec ODE ;
- refaire un run `414_no_ode` sans ODE, car l'ODE à 414 est trop lente ;
- sauvegarder les courbes dans des fichiers `.npz` ;
- superposer ensuite les courbes 128 et 414.

## 1. Paramètres importants

À 300 keV :

```text
lambda ≃ 0.0196875 Å
lambda / 2 ≃ 0.00984 Å
```

Pour la cellule Au :

```text
a = 4.076 Å
```

Donc pour avoir un pas transverse proche de `lambda/2` :

```text
N ≃ 4.076 / 0.00984 ≃ 414
```

Le pas réel avec 414 pixels est :

```text
4.076 / 414 = 0.0098454 Å
```

Le pas 128 initial est :

```text
4.076 / 128 = 0.03184375 Å
```

L'ODE ne suréchantillonne pas `x,y` : elle garde la grille transverse du notebook. Le solveur adaptatif raffine seulement l'intégration selon `z`. Dans le cas 128, le pas interne maximal estimé était environ :

```text
dz_ODE,max ≃ 0.010 Å ≃ 0.51 lambda
```

## 2. Cellule de configuration à modifier

Dans la cellule contenant :

```python
# --- Crystal parameters ---
a_central = 4.076
n_slices_per_cell = 128
energy = 300e3
n_cells_range = range(0, 101)
```

remplacer le bloc de configuration jusqu'à `ny, nx = gpts` par ceci.

Pour le run 128 avec ODE :

```python
# --- Crystal parameters ---
a_central = 4.076
n_slices_per_cell = 128
energy = 300e3
n_cells_range = range(0, 101)

# Choose the run.
run_label = "128_with_ode"
run_ode = True
requested_gpts = None

# Used only when requested_gpts is None.
pixel_size_x = 0.0318  # Angstrom
pixel_size_y = 0.0318  # Angstrom

beam_target_mrad = 135.0
beam_target_axis = "y"

atoms = bulk("Au", "fcc", a=a_central, cubic=True)
atoms.info["thermal_sigma"] = 0.0
atoms.arrays["thermal_sigma"] = np.zeros(len(atoms))

if requested_gpts is None:
    gpts, sampling = grid_from_pixel_size(atoms, pixel_size_y, pixel_size_x)
else:
    gpts = tuple(requested_gpts)  # gpts = (ny, nx)
    sampling = (
        float(atoms.get_cell()[1, 1]) / gpts[0],  # y sampling
        float(atoms.get_cell()[0, 0]) / gpts[1],  # x sampling
    )

ny, nx = gpts
```

Pour le run 414 sans ODE, changer seulement :

```python
run_label = "414_no_ode"
run_ode = False
requested_gpts = (414, 414)
```

Important : si tu définis `requested_gpts = (414, 414)`, tu n'as pas besoin de redéfinir séparément `nx, ny` ailleurs. La ligne `ny, nx = gpts` le fait.

## 3. Cellule ODE à rendre optionnelle

Remplacer la cellule ODE par :

```python
if run_ode:
    kg_ode_results = {beam_key: [] for beam_key, *_ in tracked_beams}
    w_kg = psi0
    phi_kg = None

    for i in tqdm(range(len(n_cells_range)), desc="Full KG ODE sweep"):
        if i > 0:
            w_kg, phi_kg, _, _ = simulate_kg_ode_full(
                pot_array_wk,
                w_kg,
                slice_dz,
                energy,
                sampling,
                initial_phi=phi_kg,
            )
            w_kg = jnp.array(w_kg)

        w_kg_np = np.asarray(w_kg)
        for beam_key, h, k, _ in tracked_beams:
            kg_ode_results[beam_key].append(beam_amplitude_normalized(w_kg_np, h, k))

    kg_ode_results = {key: np.array(values) for key, values in kg_ode_results.items()}
    final_summary = ", ".join(
        f"{label} final: {kg_ode_results[beam_key][-1]:.6f}"
        for beam_key, _, _, label in tracked_beams
    )
    print(f"KG ODE {final_summary}")
else:
    kg_ode_results = {
        beam_key: np.full(len(n_cells_range), np.nan)
        for beam_key, *_ in tracked_beams
    }
    w_kg = None
    phi_kg = None
    print("KG ODE skipped for this run.")
```

## 4. Sauvegarder les résultats d'un run

Ajouter une cellule juste après le sweep MS/AS/WPM et la cellule ODE :

```python
from pathlib import Path

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

data = {
    "x": np.array(list(n_cells_range), dtype=float),
    "sampling": np.array(sampling, dtype=float),
    "gpts": np.array(gpts, dtype=int),
    "cell_thickness": float(cell_thickness),
    "slice_dz": float(slice_dz),
    "energy": float(energy),
    "wavelength": float(wavelength),
    "run_ode": np.array(run_ode),
}

for key, value in beam_results.items():
    data[f"beam_{key}"] = np.asarray(value)

if run_ode:
    for beam_key, value in kg_ode_results.items():
        data[f"ode_{beam_key}"] = np.asarray(value)

out_path = results_dir / f"axel_lubk_results_{run_label}.npz"
np.savez(out_path, **data)
print(f"Saved {out_path}")
```

Cela doit produire par exemple :

```text
results/axel_lubk_results_128_with_ode.npz
results/axel_lubk_results_414_no_ode.npz
```

## 5. Superposer 128 et 414

Après avoir produit les deux fichiers `.npz`, ajouter cette cellule :

```python
from pathlib import Path

results_dir = Path("results")

runs = {
    "128 with ODE": results_dir / "axel_lubk_results_128_with_ode.npz",
    "414 no ODE": results_dir / "axel_lubk_results_414_no_ode.npz",
}

loaded = {label: np.load(path) for label, path in runs.items()}

method_keys = {
    "ms": "Fresnel MS",
    "as": "Angular Spectrum MS",
    "wpm": "WPM",
}

beam_keys = ["00"]
if any("beam_ms_target" in data.files for data in loaded.values()):
    beam_keys.append("target")

ncols = len(beam_keys)
fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 4.8), squeeze=False)
axes = axes[0]

for ax, beam_key in zip(axes, beam_keys):
    for run_label, data in loaded.items():
        x_nm = data["x"] * float(data["cell_thickness"]) * 0.1
        for method_key, method_label in method_keys.items():
            array_name = f"beam_{method_key}_{beam_key}"
            if array_name not in data.files:
                continue
            ax.plot(
                x_nm,
                data[array_name],
                label=f"{run_label} - {method_label}",
                linewidth=1.8,
            )

        ode_name = f"ode_{beam_key}"
        if ode_name in data.files:
            ax.plot(
                x_nm,
                data[ode_name],
                "k-",
                linewidth=2.6,
                alpha=0.75,
                label=f"{run_label} - Full KG ODE",
            )

    ax.set_title(f"Beam {beam_key}")
    ax.set_xlabel("Thickness (nm)")
    ax.set_ylabel("Normalized beam amplitude")
    ax.grid(True, alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.03),
    ncol=2,
    frameon=False,
    fontsize=8,
)
fig.suptitle("Axel/Lubk verification: comparison of transverse samplings", y=0.98)
fig.tight_layout(rect=[0.0, 0.12, 1.0, 0.92])
plt.show()
```

Si la légende est encore coupée :

```python
fig.savefig("comparison_128_414.png", dpi=200, bbox_inches="tight")
```

## 6. Pourquoi le faisceau diffracté est bien le même

Le faisceau cible est choisi par :

```python
beam_target = beam_for_angle_mrad(
    gpts, sampling, wavelength, beam_target_mrad, axis=beam_target_axis
)
```

L'indice du faisceau dépend essentiellement de :

```text
index = round(f_target * N * dx)
```

Or :

```text
N * dx = L
```

où `L` est la taille réelle de la cellule. Donc si la taille de cellule reste `4.076 Å`, l'indice physique du faisceau de Bragg reste le même quand on passe de 128 à 414. Ce qui change, c'est le cutoff de Nyquist : la grille 414 contient davantage de hautes fréquences.

## 7. Interprétation des différences à grande épaisseur

Les différences entre 128, 218 et 414 à forte épaisseur sont probablement dues à la convergence transverse / cutoff spectral plutôt qu'à des erreurs d'arrondi machine.

Raisons :

- le potentiel Au est très piqué ;
- une grille plus fine représente mieux les hautes fréquences ;
- ces hautes fréquences peuvent intervenir comme états intermédiaires dans la diffusion multiple ;
- les petites différences s'accumulent avec l'épaisseur.

Test recommandé :

- comparer 128, 218, 300, 414, éventuellement 512 sans ODE ;
- vérifier si 300 et 414 deviennent proches ;
- surveiller la conservation de la norme :

```python
np.sum(np.abs(wave) ** 2)
```

ou en espace réciproque :

```python
np.sum(np.abs(np.fft.fft2(wave)) ** 2)
```

