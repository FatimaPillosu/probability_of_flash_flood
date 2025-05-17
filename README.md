# Probability of Flash Floods

[![License](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey.svg)](LICENSE)  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx)  
[![Build](https://github.com/FatimaPillosu/probability_of_flash_flood/actions/workflows/ci.yml/badge.svg)](https://github.com/FatimaPillosu/probability_of_flash_flood/actions/workflows/ci.yml)

> **Probabilistic forecasting tools and datasets underpinning the PhD thesis _“Probability of Flash Floods”_.**  
> This repository contains Python modules, configuration files, and key derived datasets required to reproduce the experiments, figures, and tables in the thesis and the associated journal articles.

---

## Contents

| Folder | Description |
| ------ | ----------- |
| `code/` | Production-ready Python package (`flashprob/`) with model classes, feature engineering pipelines, and evaluation utilities. |
| `data/` | Symlinks / git-ignored placeholders for large raw ECMWF and gauge data. See **Data Access** below. |

---

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com/FatimaPillosu/probability_of_flash_flood.git
cd probability_of_flash_flood

# 2. Create environment
conda env create -f environment.yml     # or: pip install -r requirements.txt
conda activate flashprob
```
---

## Data access
Large raw datasets (ECMWF ensemble forecasts, precipitation radar composites, and gauge archives) are not stored in git. Instead we provide the data in a Zenodod repository.


## Citation
@phdthesis{Pillosu_2025,
  author = {Fatima Maria Pillosu},
  title  = {Probability of Flash Floods},
  school = {University of Reading},
  year   = {2025},
  doi    = {10.1234/example.doi}
}

@software{Pillosu_FlashProb_2025,
  author  = {Pillosu, Fatima Maria},
  title   = {Code and Data for the PhD thesis 'Probability of Flash Floods'},
  version = {0.0.1},
  doi     = {10.5281/zenodo.xxxxxxx},
  url     = {https://github.com/FatimaPillosu/probability_of_flash_flood},
  date    = {2025-05-14}
}


## Contributing
Pull requests are welcome! To keep the history tidy:
1. Open an issue first to discuss major changes.
2. Create a feature branch off main.
3. Ensure pre-commit hooks pass (black, ruff, mypy).
4. Add or update unit tests in tests/.
5. Submit the PR and enjoy automatic CI feedback (GitHub Actions).


## License
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


## Contact and support
Email: fatima.pillosu@ecmwf.int
ORCID: 0000-0001-8127-0990
Research Group: Reading Meteorology & ECMWF