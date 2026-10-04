# Manuscript figures

Data figures for the paper are regenerated locally by `scripts/report/build_all.py`
into `results/paper/`. They are not part of the repository or the data deposit.

The system diagram is a local file, `figures/fig1_system.svg`, and is not tracked.
When that file is present, the build copies it into `results/paper/`. When it is
absent, the data figures still build and the diagram step is skipped.
