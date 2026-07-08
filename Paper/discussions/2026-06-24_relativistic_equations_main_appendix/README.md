# Discussion — Équations relativistes, indice électronique et structure du papier

Date : 2026-06-24  
Mots-clés : Schrödinger relativiste corrigée, Klein-Gordon, \(E_0\), \(\gamma\), \(p_0\), \(n_S\), \(\sigma\), main.tex, appendix

## Relations relativistes retenues

Les relations cohérentes à introduire sont :

\[
E_0=m_0c^2,
\]

\[
E_{\mathrm{tot}}
=
E_0+E_{\mathrm{kin}}
=
\gamma E_0,
\]

\[
E_{\mathrm{tot}}^2
=
p_0^2c^2+E_0^2,
\]

\[
\gamma
=
\frac{1}{\sqrt{1-v^2/c^2}},
\]

\[
p_0
=
\gamma m_0 v
=
\frac{h}{\lambda}
=
h k_0.
\]

Attention à la notation :

- si \(m=m_0\), alors \(p_0=\gamma m_0v\) ;
- si \(m=\gamma m_0\) est appelé masse relativiste, alors \(p_0=mv\) ;
- ne pas écrire \(m\gamma v\) si \(m\) désigne déjà la masse relativiste.

## Équation stationnaire corrigée

Forme discutée pour la section 2.1 :

\[
\left[
-\frac{\hbar^2}{2\gamma m_0}\nabla^2
+
E_{\mathrm{pot}}(\mathbf r)
\right]\psi(\mathbf r)
=
\frac{p_0^2}{2\gamma m_0}
\psi(\mathbf r).
\]

Avec :

\[
E_{\mathrm{pot}}(\mathbf r)=-eV(\mathbf r),
\]

où \(V\) est le potentiel électrostatique du spécimen en volts et \(e>0\).

## Subtilité sur le membre de droite

Le membre de droite :

\[
\frac{p_0^2}{2\gamma m_0}
\]

ressemble à une énergie cinétique non relativiste avec la masse relativiste \(\gamma m_0\), mais ce n’est ni \(E_{\mathrm{tot}}\), ni exactement \(E_{\mathrm{kin}}\).

On a :

\[
\frac{p_0^2}{2\gamma m_0}
=
\frac{E_{\mathrm{tot}}^2-E_0^2}{2E_{\mathrm{tot}}}
=
E_{\mathrm{kin}}
\frac{E_{\mathrm{tot}}+E_0}{2E_{\mathrm{tot}}}.
\]

La dépendance temporelle physique reste :

\[
\Psi(\mathbf r,t)
=
\psi(\mathbf r)
\exp\left(-\frac{iE_{\mathrm{tot}}t}{\hbar}\right).
\]

Le facteur \(1/2\) apparaît lorsqu’on divise l’équation relativiste linéarisée par \(2E_{\mathrm{tot}}\) pour obtenir une forme de type Schrödinger.

## Remarque LaTeX proposée pour l’appendice

```latex
\subsection{A Note on the Energy Eigenvalue in the Corrected Schrödinger Equation}
\label{app:corrected_schrodinger_eigenvalue}

The right-hand side of the corrected Schrödinger equation,
\[
\frac{p_0^2}{2\gamma m_0},
\]
has the formal appearance of a nonrelativistic kinetic energy written with the
relativistic mass \(\gamma m_0\). This quantity should not, however, be
identified with the total relativistic energy \(E_{\mathrm{tot}}\), nor exactly
with the kinetic energy \(E_{\mathrm{kin}}\). Using
\[
E_{\mathrm{tot}}^2=p_0^2c^2+E_0^2,
\qquad
\gamma m_0=\frac{E_{\mathrm{tot}}}{c^2},
\]
one obtains
\[
\frac{p_0^2}{2\gamma m_0}
=
\frac{E_{\mathrm{tot}}^2-E_0^2}{2E_{\mathrm{tot}}}
=
E_{\mathrm{kin}}
\frac{E_{\mathrm{tot}}+E_0}{2E_{\mathrm{tot}}}.
\]
The factor \(1/2\) therefore originates from the Schrödinger-like
normalization of the relativistic wave equation. The physical time dependence
of the stationary state remains governed by \(E_{\mathrm{tot}}\),
\[
\Psi(\mathbf r,t)
=
\psi(\mathbf r)\exp(-iE_{\mathrm{tot}}t/\hbar).
\]
```

## Indice électronique linéarisé

La forme Helmholtz discutée est :

\[
\nabla^2\psi
+
(2\pi k_0)^2 n^2(\mathbf r)\psi
=
0.
\]

Dans l’approximation linéarisée adaptée à la microscopie électronique :

\[
n_S^2(\mathbf r)
=
1
-
\frac{2\gamma m_0E_{\mathrm{pot}}(\mathbf r)}
{h^2k_0^2}.
\]

Comme \(E_{\mathrm{pot}}=-eV\), cela devient :

\[
n_S^2(\mathbf r)
=
1+
\frac{\sigma}{\pi k_0}V(\mathbf r).
\]

Constante d’interaction :

\[
\sigma
=
\frac{2\pi\gamma m_0e}{h^2k_0}
=
\frac{2\pi\gamma m_0e\lambda}{h^2}.
\]

Au premier ordre, si \(n_S=1+\delta n\) :

\[
\delta n
=
\frac{\sigma V}{2\pi k_0}.
\]

## Structure rédactionnelle proposée pour `main.tex`

La section 2.1 devait être simplifiée ainsi :

1. Commencer par Fujiwara et l’appendice.
2. Donner l’équation stationnaire corrigée.
3. Donner les relations relativistes \(E_0\), \(E_{\mathrm{tot}}\), \(p_0\), \(\gamma\), \(\lambda\), \(k_0\).
4. Réécrire en forme Helmholtz.
5. Expliquer que l’indice peut être traité en régime KG plus général ou en approximation linéarisée \(n_S\).
6. Dans le texte principal, ne donner que \(n_S\) et introduire immédiatement \(\sigma\).

