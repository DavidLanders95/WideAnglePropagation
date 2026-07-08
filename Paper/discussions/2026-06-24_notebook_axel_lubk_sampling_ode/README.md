# Discussion — Notebook Axel/Lubk, sampling, ODE et comparaison de convergence

Date : 2026-06-24  
Mots-clés : notebook Axel Lubk, sampling, ODE, 128, 414, AS, MS, WPM, convergence transverse

## Notebook concerné

Notebook actuel :

```text
C:\Users\jr122738\Documents\New project\WideAnglePropagation\notebooks\verification\01_axel_lubk_verification.ipynb
```

Ancienne version repérée :

```text
C:\Users\jr122738\Documents\New project\WideAnglePropagation\archive\au_axel_lubk_verification.ipynb
```

## Pas de grille \(128\times128\)

Dans le notebook actuel :

```python
a_central = 4.076
n_slices_per_cell = 128
pixel_size_x = 0.0318  # Angstrom
pixel_size_y = 0.0318  # Angstrom
```

Le pas effectif est :

\[
\Delta x=\Delta y=\frac{4.076}{128}
=0.03184375\ \text{\AA}.
\]

Comme il y a aussi 128 slices par cellule :

\[
\Delta z_{\mathrm{slice}}
=
\frac{4.076}{128}
=0.03184375\ \text{\AA}.
\]

## Pas interne de l’ODE

L’ODE utilise la même grille transverse :

\[
\Delta x=\Delta y=0.03184375\ \text{\AA}.
\]

Elle ne suréchantillonne pas \(x,y\). Elle raffine seulement l’intégration en \(z\).

Dans `simulate_kg_ode_full`, le solveur Diffrax utilise un pas interne adaptatif en \(z\), techniquement appelé `dt` par le solveur, mais physiquement :

\[
dt_{\mathrm{solver}}\equiv dz_{\mathrm{ODE}}.
\]

La borne maximale imposée est :

\[
dz_{\max}
=
\frac{3.5}
{\sqrt{
\max(k_\perp^2)
+
\max(|k_0^2 n^2|)
}}.
\]

À 300 keV, en approximation vide :

\[
\lambda\simeq0.0196875\ \text{\AA},
\]

\[
k_0=\frac{2\pi}{\lambda}\simeq319.15\ \text{\AA}^{-1}.
\]

Pour la grille \(128\times128\), on avait estimé :

\[
dz_{\max}\simeq0.01005\ \text{\AA}.
\]

Donc :

\[
dz_{\mathrm{ODE,internal}}\lesssim0.010\ \text{\AA}.
\]

C’est environ :

\[
0.51\,\lambda.
\]

La valeur minimale réelle n’est pas connue a priori et n’est pas sauvegardée par le code actuel ; elle dépend du contrôleur adaptatif et des tolérances :

```python
rtol = 1e-8
atol = 1e-10
```

## Grille \(414\times414\)

Pour utiliser un pas transverse proche de \(\lambda/2\) à 300 keV :

\[
\lambda/2\simeq0.00984\ \text{\AA}.
\]

Avec la cellule Au :

\[
N\simeq\frac{4.076}{0.00984}\simeq414.
\]

Modification conseillée :

```python
gpts = (414, 414)
sampling = (
    float(atoms.get_cell()[0, 0]) / gpts[0],
    float(atoms.get_cell()[1, 1]) / gpts[1],
)
ny, nx = gpts
```

ou :

```python
pixel_size_x = 4.076 / 414
pixel_size_y = 4.076 / 414
gpts, sampling = grid_from_pixel_size(atoms, pixel_size_y, pixel_size_x)
ny, nx = gpts
```

Le pas effectif devient :

\[
\Delta x=\Delta y=\frac{4.076}{414}
\simeq0.0098454\ \text{\AA}.
\]

## Pourquoi l’ODE devient trop longue à 414

Le passage de \(128\) à \(414\) augmente le nombre de pixels transverses d’environ :

\[
\left(\frac{414}{128}\right)^2\simeq10.5.
\]

Mais pour l’ODE, c’est pire que cela, car :

- il y a plus de degrés de liberté transverses ;
- \(k_{\perp,\max}\) augmente ;
- le pas interne maximal en \(z\) diminue ;
- le solveur adaptatif doit intégrer l’oscillation rapide de l’équation complète.

Conclusion pratique : arrêter le run ODE \(414\times414\) était raisonnable.

## Comparer MS/AS/WPM sans ODE

On peut sauter la cellule :

```python
kg_ode_results = {beam_key: [] for beam_key, *_ in tracked_beams}
w_kg = psi0
phi_kg = None
...
```

et ne conserver que :

```python
methods = ("ms", "as", "wpm")
```

Pour sauvegarder les résultats \(128\) avec ODE :

```python
tag = "128_with_ode"

data = {
    "x": np.array(list(n_cells_range), dtype=float),
    "sampling": np.array(sampling),
    "cell_thickness": float(cell_thickness),
}

for key, value in beam_results.items():
    data[f"beam_{key}"] = np.asarray(value)

for beam_key, value in kg_ode_results.items():
    data[f"ode_{beam_key}"] = np.asarray(value)

np.savez(f"axel_lubk_results_{tag}.npz", **data)
```

Pour sauvegarder \(414\) sans ODE :

```python
tag = "414_no_ode"

data = {
    "x": np.array(list(n_cells_range), dtype=float),
    "sampling": np.array(sampling),
    "cell_thickness": float(cell_thickness),
}

for key, value in beam_results.items():
    data[f"beam_{key}"] = np.asarray(value)

np.savez(f"axel_lubk_results_{tag}.npz", **data)
```

## Pourquoi le même faisceau diffracté est comparé

Le faisceau cible est choisi via :

```python
beam_target = beam_for_angle_mrad(
    gpts, sampling, wavelength, beam_target_mrad, axis=beam_target_axis
)
```

La fonction calcule essentiellement :

\[
\mathrm{index}
=
\mathrm{round}
\left(
f_{\mathrm{target}}N\Delta x
\right).
\]

Or :

\[
N\Delta x=L,
\]

la taille réelle de la cellule.

Donc :

\[
\mathrm{index}
=
\mathrm{round}
\left(
\frac{\theta_{\mathrm{target}}}{\lambda}L
\right).
\]

Si \(L=4.076\) Å est conservé, l’indice du faisceau diffracté reste le même malgré le changement de \(N\).

Ce qui change est le cutoff de Nyquist :

\[
f_{\max}=\frac{1}{2\Delta x}.
\]

Donc \(414\) contient plus de hautes fréquences, mais les pics de Bragg bas ordre restent aux mêmes indices physiques.

## Différences fortes à grande épaisseur

Les différences entre \(128\), \(218\), \(414\) à grande épaisseur sont probablement dues à la convergence transverse / au cutoff spectral, pas à des erreurs d’arrondi machine.

Le code AS et WPM traite les évanescentes avec une racine complexe :

\[
k_z=\sqrt{k_0^2-k_\perp^2}
\]

ou :

\[
k_z=\sqrt{n^2k_0^2-k_\perp^2}.
\]

Donc les composantes évanescentes sont amorties plutôt que propagées comme des ondes réelles.

Cause la plus probable :

- le potentiel atomique Au est très piqué ;
- une grille plus fine représente mieux les hautes fréquences ;
- ces hautes fréquences peuvent agir comme états intermédiaires dans la diffusion multiple ;
- les petites différences s’accumulent à grande épaisseur.

Test recommandé :

- comparer \(128\), \(218\), \(300\), \(414\), éventuellement \(512\) sans ODE ;
- vérifier si \(300\) et \(414\) convergent ;
- vérifier la conservation de la norme :

\[
\sum_{x,y}|\psi(x,y)|^2
\]

ou :

\[
\sum_g|\Psi_g|^2.
\]

