# Discussion — Brenner, BPM, WPM et analogie multislice

Date : 2026-06-24  
Mots-clés : Brenner, Singer, BPM, WPM, angular spectrum, multislice, \(\delta n\), \(\sigma V\), correction angulaire

## Papier discuté

Article placé dans :

```text
C:\Users\jr122738\Documents\New project\docs\brenner_ao-32-26-4984.pdf
```

Référence discutée : Brenner & Singer, *Light propagation through microlenses: a new simulation method*, Applied Optics 32, 4984 (1993).

## Équation (4) de Brenner

Brenner part de la conservation de la composante transverse du vecteur d’onde, équivalente à la loi de Snell-Descartes :

\[
n\sin\vartheta = \text{constante}.
\]

Dans leur notation :

\[
\lambda_0\nu_x
=
\bar n(z)\sin\bar\vartheta
=
[\bar n(z)+\delta n(x)]\sin(\bar\vartheta+\delta\vartheta).
\]

Développement au premier ordre :

\[
[\bar n+\delta n]\sin(\bar\vartheta+\delta\vartheta)
\simeq
\bar n\sin\bar\vartheta
+\delta n\sin\bar\vartheta
+\bar n\,\delta\vartheta\cos\bar\vartheta.
\]

Comme la quantité doit rester égale à \(\bar n\sin\bar\vartheta\), on obtient :

\[
\delta n\sin\bar\vartheta
+\bar n\,\delta\vartheta\cos\bar\vartheta
\simeq 0.
\]

D’où :

\[
\delta\vartheta
\simeq
-\frac{\delta n}{\bar n}\tan\bar\vartheta.
\]

Brenner écrit l’amplitude de l’erreur angulaire :

\[
\delta\vartheta
\simeq
\frac{\delta n(x)}{\bar n(z)}\tan\bar\vartheta.
\]

Le signe dépend de la convention choisie pour \(\delta\vartheta\). Physiquement, si \(n\) augmente localement et si \(k_x\) est conservé, l’angle par rapport à \(z\) diminue.

## Phase factor dépendant de l’angle

Pour une onde plane dans un milieu d’indice \(n\) :

\[
k = n k_0,
\qquad
k_0=\frac{2\pi}{\lambda_0}.
\]

Si l’onde fait un angle \(\vartheta\) avec l’axe \(z\), sa composante longitudinale est :

\[
k_z = n k_0\cos\vartheta.
\]

Le facteur de phase pour une propagation axiale \(\delta z\) est donc :

\[
\exp(-i k_z\delta z)
=
\exp[-i n k_0\cos\vartheta\,\delta z].
\]

Dans le BPM standard discuté par Brenner, la correction d’indice locale est appliquée en espace réel comme :

\[
\exp[-i\delta n(x)k_0\delta z],
\]

sans facteur \(\cos\vartheta\). Cela revient à appliquer la correction comme si toutes les composantes traversaient la tranche normalement.

L’erreur angulaire de phase est donc proportionnelle à :

\[
1-\cos\vartheta.
\]

Plus précisément :

\[
\Delta\phi
=
\delta n\,k_0\,\delta z\left(1-\cos\vartheta\right).
\]

## Correction importante sur le BPM de Brenner

Il ne faut pas assimiler trop vite le BPM de Brenner à un Fresnel-MS paraxial.

L’équation (2) de Brenner contient :

\[
\exp\left[
i\delta z
\sqrt{\bar n^2(z)k_0^2-(2\pi\nu_x)^2}
\right].
\]

Donc la propagation homogène est de type angular spectrum dans un milieu d’indice moyen \(\bar n\), pas une simple approximation de Fresnel.

Leur BPM peut être vu comme :

\[
\text{BPM Brenner}
=
\text{propagation AS homogène avec indice moyen}
+
\text{phase grating locale indépendante de l’angle}.
\]

L’erreur critiquée par Brenner n’est pas nécessairement la propagation libre paraxiale ; elle vient surtout du fait que l’interaction locale avec \(\delta n(x)\) est appliquée sans dépendance à l’angle.

## Analogie avec le multislice électronique

En optique :

\[
t_{\mathrm{opt}}(x)
=
\exp[-i k_0\delta n(x)\Delta z].
\]

En multislice électronique :

\[
t_{\mathrm{el}}(\mathbf r_\perp)
=
\exp[i\sigma V(\mathbf r_\perp)\Delta z].
\]

La correspondance est :

\[
\sigma V
\longleftrightarrow
k_0\delta n
\]

si \(k_0=2\pi/\lambda\). Avec la convention du papier où \(k_0=1/\lambda\), il faut écrire :

\[
\sigma V
=
2\pi k_0\,\delta n.
\]

Donc :

\[
\delta n
=
\frac{\sigma V}{2\pi k_0}.
\]

Cette relation est cohérente avec :

\[
n_S^2
=
1+\frac{\sigma}{\pi k_0}V.
\]

Si \(n_S=1+\delta n\), alors au premier ordre :

\[
n_S^2\simeq 1+2\delta n,
\]

d’où :

\[
\delta n
=
\frac{\sigma V}{2\pi k_0}.
\]

## Point clé pour la discussion du papier

Le facteur de phase multislice standard :

\[
\exp[i\sigma V(\mathbf r_\perp)\Delta z]
\]

n’introduit pas explicitement de correction d’angle.

Il agit en espace réel et applique le même déphasage local à toutes les composantes angulaires de l’onde. Il ne contient ni \(\cos\theta\), ni \(k_z\), ni une dépendance explicite à la direction incidente.

Même si AS-MS améliore la propagation homogène par :

\[
k_z=\sqrt{K_0^2-K_\perp^2},
\]

l’étape d’interaction locale reste une phase grating indépendante de l’angle.

## Formulation anglaise proposée pour la discussion

```latex
Standard multislice can be viewed as the electron-optical analogue of a
split-step beam-propagation method. The wavefield is alternately propagated
through a homogeneous slice in reciprocal space and multiplied by a local
transmission function in real space. The electron-optical phase grating
\[
\exp[i\sigma V(\mathbf r_\perp)\Delta z]
\]
is directly analogous to the optical phase factor associated with a local
refractive-index perturbation.

This analogy makes clear that the standard transmission function is
angle-independent. The same projected-potential phase shift is applied to all
plane-wave components of the wavefield, irrespective of their incidence angle.
Thus, although an angular-spectrum propagator can describe the homogeneous
propagation step beyond the Fresnel approximation, the interaction step still
neglects angular corrections associated with the longitudinal wave vector
\(k_z\), or equivalently with factors such as \(\cos\theta\).
```

