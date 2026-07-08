# Figure Sources

Standalone LaTeX/TikZ sources used to generate paper figures live here.
Generated figure files that are included by `Paper/main.tex` stay under
`Paper/figures/`.

The current method-explainer sources are in `method_explainer_generations/`.
Build an individual source with `latexmk -pdf <file>.tex` from that directory,
then copy or move the generated paper-ready PDF into `Paper/figures/` if it is
used by the manuscript.
