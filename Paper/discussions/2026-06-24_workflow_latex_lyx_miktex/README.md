# Discussion — Workflow LaTeX, LyX, VS Code et MiKTeX

Date : 2026-06-24  
Mots-clés : LaTeX, LyX, VS Code, MiKTeX, formules, outline, installation

## Problème identifié

Difficultés exprimées :

1. Manque de repères clairs dans `main.tex` / chapitres / sections.
2. Lecture difficile des formules en LaTeX brut.
3. Envie de retrouver un environnement proche de LyX, plus visuel.

## VS Code peut aider, mais ne remplace pas totalement LyX

Extensions et fonctions utiles :

- LaTeX Workshop ;
- panneau Outline ;
- repliage des sections ;
- aperçu PDF côte à côte ;
- synchronisation source/PDF ;
- aperçu des équations au survol ;
- breadcrumbs.

Mais VS Code reste essentiellement un éditeur de code. Il n’a pas le confort WYSIWYM de LyX.

## LyX comme option raisonnable

LyX est adapté si l’objectif est :

- voir les titres clairement ;
- écrire dans une structure logique ;
- voir les équations rendues ;
- réduire la charge cognitive du LaTeX brut.

Prudence :

- importer un gros projet LaTeX existant dans LyX peut être fragile ;
- le projet utilise probablement des packages, macros, figures, bibliographie, appendices ;
- il vaut mieux tester LyX sur un petit document d’abord.

Stratégie recommandée :

- garder `main.tex` comme source officielle ;
- utiliser LyX pour écrire/réécrire des morceaux si utile ;
- exporter ou copier proprement vers `main.tex` ;
- garder VS Code/Codex pour compiler, versionner, corriger.

## Installation LyX

Page officielle :

```text
https://www.lyx.org/Download
```

Installateur Windows actuel repéré :

```text
LyX-251-Installer-1-x64.exe
```

MiKTeX est déjà installé en mode utilisateur :

```text
C:\Users\jr122738\AppData\Local\Programs\MiKTeX\miktex\bin\x64\miktex-console.exe
```

Si l’installateur LyX demande les droits admin pour `C:\Program Files`, essayer une installation utilisateur :

```text
C:\Users\jr122738\AppData\Local\Programs\LyX
```

ou :

```text
C:\Users\jr122738\Programs\LyX
```

## MiKTeX et installation automatique des packages

Dans MiKTeX Console, la bonne option est dans :

```text
Settings > General > Package installation
```

La capture d’écran montrait :

```text
You can choose whether missing packages are to be installed automatically (on-the-fly):

Always
Ask me
Never
```

Recommandation :

```text
Always
```

L’option “For anyone who uses this computer” peut être grisée si MiKTeX est lancé en mode utilisateur sans droits admin. Ce n’est pas bloquant.

Après modification :

1. Lancer LyX.
2. Faire :

```text
Tools > Reconfigure
```

3. Redémarrer LyX.
4. Tester un petit document avec une équation.

## Remarque LyX + MiKTeX

LyX + MiKTeX a parfois été capricieux sous Windows :

- chemins non détectés ;
- packages installés à la demande ;
- fenêtres de confirmation MiKTeX ;
- besoin de reconfigurer LyX après mise à jour MiKTeX.

Alternative plus robuste mais plus lourde :

```text
TeX Live
```

Avantage TeX Live :

- installation plus complète ;
- moins d’interactions à la demande ;
- souvent très stable avec LyX.

Inconvénient :

- beaucoup plus gros.

Conseil final :

1. Tester LyX + MiKTeX puisque MiKTeX est déjà là.
2. Si cela devient pénible, envisager TeX Live.
3. Éviter d’avoir MiKTeX et TeX Live actifs simultanément dans le `PATH`.

