# Fork Information

This document describes how this fork relates to its upstream project,
[the Tractor](https://github.com/dstndstn/tractor): what changed, why it exists,
and how it is kept in sync. For general usage, see [`README.md`](README.md).

## Upstream

- **Original project:** [dstndstn/tractor](https://github.com/dstndstn/ "the Tractor: measuring astronomical sources via probabilistic inference"
- **Original license:** GPLv2 (unchanged in this fork -- see [`LICENSE`](LICENSE)
  and [`COPYING`](COPYING))
- **Upstream authors:** Copyright 2011-2023 Dustin Lang (Perimeter Institute)
  & David W. Hogg (NYU / Flatiron)
- **Upstream docs:** <https://thetractor.org/doc> and
  <https://thetractor.readthedocs.org/en/latest/>

## Fork Point

This fork currently branches from the upstream at:

- **Commit:** [`8dc2bd8bce29db73968b1edf99a23b1c9130d074`](https://github.com/dstndstn/tractor/commit/8dc2bd8bce29db73968b1edf99a23b1c9130d074)
- **Tag:** No tag exists for the fork point; [`dr10.6`](https://github.com/dstndstn/tractor/releases/tag/dr10.6) was the most recent preceding tag
- **Date:** `2025-10-06`

Knowing the branch point makes divergence auditable and helps when rebasing or
merging future upstream changes. (The upstream's default branch is `main`.)

## Why This Fork Exists

This fork was created to support features and performance needed for the
forced photometry performed as part of the [NASA SPHEREx mission](https://spherex.caltech.edu)
pipelines and associated IRSA tools.
The code was developed as part of the pipeline development by the
SPHEREx Science Data Center (SSDC) at Caltech/IPAC.

Upstreaming of some of the changes is under discussion; other changes are
SPHEREx-specific code which will not be upstreamed, but because of its heritage
in `tractor` with its GPLv2 license, cannot be held in the differently-licensed
SPHEREx pipelines repositories.

## What Differs From Upstream

- `pyproject.toml` and `requirements.txt`: added dependencies for `pip` installation
- `tractor/lsqr_optimizer.py`: memory optimized derivative handling
- `tractor/optimize.py`: time optimized image addition; computed covariant errors
- `tractor/psf.py`: `numba` optimized Lanczos shifting
- `tractor/spherex_utils.py`: created new module for SPHEREx-specific utilities; added function to support arbitrary PSF oversampling when fitting shapes

## Sync Policy

We import changes from the upstream into the fork periodically as part of
preparation for major SPHEREx processing cycles.

For maintainers, keeping the upstream as a second remote makes syncing straightforward:

```sh
git remote add upstream https://github.com/dstndstn/tractor.git
git fetch upstream
git merge upstream/main        # or: git rebase upstream/main
```

## License & Modifications Notice

Copyright for modifications in this fork:

```
Copyright (C) 2023-2026 California Institute of Technology
```

This fork remains licensed under the GNU General Public License, version 2,
the same terms as the original (see [`LICENSE`](LICENSE) and [`COPYING`](COPYING)).

Note that the upstream's `LICENSE` and `COPYING` files contain only the verbatim
stock GPLv2 text - they do **not** carry the project's own copyright statement.
The Tractor's actual copyright notice lives solely in the "authors & license"
section of the upstream's `README.md`:

> Copyright 2011-2023 Dustin Lang (Perimeter Institute) & David W. Hogg (NYU / Flatiron)

That section is therefore the canonical copyright notice that GPLv2 requires
to be kept intact. This fork preserves it verbatim: the fork notice is prepended
*above* the upstream's content in `README.md` precisely so that block is never
edited, moved, or removed. (The license is stated only as "GPLv2", with no
"or later" clause in the README, the per-file headers, or the license files, so
this fork treats it as GPLv2 specifically.)

As required by the GPL, modified files carry a notice indicating that they were
changed and when; the full record of changes is in [`CHANGELOG.md`](CHANGELOG.md).

## Contact

Questions about the SPHEREx pipelines should be directed to the
IRSA help-desk (https://irsa.ipac.caltech.edu/docs/help_desk.html).
