# LaTeX Paper Skeleton (Project 3)

This folder contains a base IEEE conference paper structure aligned with Project 3 requirements:
- Mandatory sections: Abstract, Introduction, ROCKET method, Experimentation, Results, Conclusions
- IEEE format (`IEEEtran`)
- Target length: max 5 pages

## Files
- `main.tex`: paper entry point
- `sections/*.tex`: section-level content
- `references.bib`: bibliography
- `figures/`: place generated figures here

## Build (example)

```bash
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Integrating project outputs
Copy or link generated assets from `results/` to `latex/figures/` (for example: `accuracy_heatmap.pdf`, `cd_diagram.pdf`, etc.) and replace placeholder text/tables with your final analysis.
