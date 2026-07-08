# Discussion — Helmholtz, Huygens et multislice revisité

Date : 2026-06-24  
Mots-clés : Helmholtz, Huygens, Green, Kirchhoff, multislice, convolution, angular spectrum

## Solution de Helmholtz dans le vide

L’équation de Helmholtz homogène dans le vide est :

\[
(\nabla^2+K_0^2)\psi(\mathbf r)=0.
\]

Une onde plane est :

\[
\psi(\mathbf r)=\exp(i\mathbf K\cdot\mathbf r),
\qquad
|\mathbf K|=K_0.
\]

Une onde sphérique sortante est liée à la fonction de Green :

\[
G(\mathbf r,\mathbf r')
=
\frac{e^{iK_0|\mathbf r-\mathbf r'|}}
{4\pi|\mathbf r-\mathbf r'|}.
\]

Elle vérifie :

\[
(\nabla^2+K_0^2)G(\mathbf r,\mathbf r')
=
-\delta(\mathbf r-\mathbf r').
\]

En dehors de la source, elle vérifie l’équation homogène.

## Ondes planes et ondes sphériques

Une onde sphérique peut être reconstruite comme une superposition continue d’ondes planes.

Les ondes planes et les ondes sphériques ne sont donc pas deux familles incompatibles ; ce sont deux représentations complémentaires :

- onde plane : direction de propagation bien définie ;
- onde sphérique : source ponctuelle / émission dans toutes les directions ;
- angular spectrum : décomposition en directions ;
- Huygens : reconstruction par sources secondaires.

## La formule qui justifie le mieux Huygens

La formulation mathématique propre du principe de Huygens est la représentation intégrale de Kirchhoff-Helmholtz :

\[
\psi(\mathbf r)
=
\int_S
\left[
G(\mathbf r,\mathbf r')
\frac{\partial \psi(\mathbf r')}{\partial n'}
-
\psi(\mathbf r')
\frac{\partial G(\mathbf r,\mathbf r')}{\partial n'}
\right]
dS'.
\]

Le terme :

\[
G\,\partial_{n'}\psi
\]

ressemble à une couche de sources monopôles : chaque point émet une onde sphérique pondérée par la dérivée normale du champ.

Le terme :

\[
-\psi\,\partial_{n'}G
\]

ressemble à une couche de dipôles : il encode la directivité / facteur d’obliquité.

Donc le Huygens naïf “chaque point réémet une onde sphérique proportionnelle au champ” est une approximation. La formule exacte demande à la fois \(\psi\) et \(\partial_n\psi\).

Sous approximations de propagation vers l’avant :

\[
\partial_n\psi\simeq iK_0\cos\alpha\,\psi,
\]

et :

\[
\partial_nG\simeq -iK_0\cos\beta\,G.
\]

Les deux termes deviennent alors proportionnels à \(\psi G\), avec des facteurs angulaires. C’est ainsi qu’apparaît la formule Huygens-Fresnel avec facteur d’obliquité.

## Huygens revisité pour le multislice

L’étape multislice standard s’écrit :

\[
\psi_{j+1}(\mathbf r_\perp)
=
\mathcal P_{\Delta z}
\left[
t_j(\mathbf r_\perp)\psi_j(\mathbf r_\perp)
\right],
\]

avec :

\[
t_j(\mathbf r_\perp)
=
\exp[i\sigma V_j(\mathbf r_\perp)\Delta z].
\]

En espace réel, la propagation \(\mathcal P_{\Delta z}\) peut être vue comme une convolution :

\[
\psi_{j+1}(\mathbf r_\perp)
=
\int
h_{\Delta z}(\mathbf r_\perp-\mathbf r'_\perp)
\,
t_j(\mathbf r'_\perp)
\psi_j(\mathbf r'_\perp)
\,d\mathbf r'_\perp.
\]

En posant :

\[
\psi_j^+(\mathbf r'_\perp)
=
t_j(\mathbf r'_\perp)\psi_j(\mathbf r'_\perp),
\]

on obtient :

\[
\psi_{j+1}(\mathbf r_\perp)
=
\int
h_{\Delta z}(\mathbf r_\perp-\mathbf r'_\perp)
\psi_j^+(\mathbf r'_\perp)
\,d\mathbf r'_\perp.
\]

Interprétation :

- chaque pixel de la slice sortante est une source secondaire cohérente ;
- il émet une ondelette vers tous les pixels de la slice suivante ;
- toutes les contributions interfèrent pour former le front suivant ;
- les directions accessibles sont limitées par la grille numérique.

Formulation synthétique :

```latex
In real space, the multislice propagation step can be interpreted as a
discrete Huygens construction: after local interaction with the projected
potential, each pixel of the exit surface of a slice acts as a secondary
coherent emitter. The field at the next slice is obtained by summing the
contributions emitted from all pixels, with a propagation kernel determined by
the chosen free-space propagator.
```

## Différence Fresnel / Angular Spectrum dans cette image

Pour Fresnel-MS :

\[
H_F(\mathbf k_\perp)
=
\exp[-i\pi\lambda\Delta z|\mathbf k_\perp|^2].
\]

Pour AS-MS :

\[
H_{\mathrm{AS}}(\mathbf K_\perp)
=
\exp\left[
i\Delta z\sqrt{K_0^2-|\mathbf K_\perp|^2}
\right].
\]

Donc :

- Fresnel-MS : ondelettes secondaires paraxiales ;
- AS-MS : ondelettes secondaires avec relation homogène exacte \(K_z=\sqrt{K_0^2-K_\perp^2}\) ;
- mais dans les deux cas, la source secondaire \(t_j=\exp(i\sigma V\Delta z)\) reste indépendante de l’angle.

Cela prépare naturellement l’interprétation de WPM :

\[
\text{WPM}
=
\text{Huygens revisité où l’émission dépend aussi de la direction}.
\]

