# LyX working folder

This folder contains a LyX working copy of the current `../main.tex`.

Use these files for structural editing:

```text
main.lyx
appendix_relativistic_hamiltonian.lyx
```

Important notes:

- This is a working/imported LyX document, not the canonical source yet.
- The canonical LaTeX source remains `../main.tex`.
- The canonical appendix source remains `../appendix_relativistic_hamiltonian.tex`.
- The title/authors/frontmatter have been simplified during import; this is intentional for now.
- The file is meant for reading, reorganising sections, editing paragraphs, and checking equations.
- When the text structure is satisfactory, export from LyX to LaTeX and we will merge the useful changes back into the canonical `.tex` files.

Validation done:

- `main.lyx` imports and opens as a LyX document.
- It exports to `main_exported.tex`.
- `main_exported.tex` compiles to `main_exported.pdf`.
- Figure names with underscores were repaired after import.
- `appendix_relativistic_hamiltonian.lyx` imports and opens as a LyX document.
- It exports to `appendix_relativistic_hamiltonian_exported.tex`.
- `appendix_relativistic_hamiltonian_exported.tex` compiles to `appendix_relativistic_hamiltonian_exported.pdf`.
- Accented characters and dashes in the appendix were checked as UTF-8 after import.

Known limitations:

- Appendix references to `app:relativistic_hamiltonian` remain unresolved because the current `main.tex` does not include the appendix file.
- The standalone appendix export has one expected unresolved reference to `sec:schrodinger_helmholtz`, because that label lives in the main paper.
- Bibliography warnings remain mostly inherited from the BibTeX entries/style and are not blocking for structural editing.
- The exported LaTeX from LyX is not expected to be identical to `../main.tex`; use it as an editing intermediate.
