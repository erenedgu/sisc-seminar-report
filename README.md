 # SISC Seminar Report


This repository contains a LaTeX-based seminar report on soft sensors for measuring energy consumption in the context of topographic uncertainty propagation in flood simulations. The work is organized as a structured academic report and supporting scripts.



<a href="https://raw.githubusercontent.com/erenedgu/sisc-seminar-report/pdf-tectonic/thesis.pdf">

<img src="https://img.shields.io/badge/View-Report_(Tectonic)-red?style=flat-square&logo=adobeacrobatreader&logoColor=white" alt="View the seminar report PDF (Tectonic)"/>

</a>

<a href="https://raw.githubusercontent.com/erenedgu/sisc-seminar-report/pdf-tinytex/thesis.pdf">

<img src="https://img.shields.io/badge/View-Report_(TinyTeX)-red?style=flat-square&logo=adobeacrobatreader&logoColor=white" alt="View the seminar report PDF (TinyTeX)"/>

</a>

<a href="https://github.com/erenedgu/sisc-seminar-report/actions/workflows/compat.yml">

<img src="https://img.shields.io/github/actions/workflow/status/erenedgu/sisc-seminar-report/compat.yml?label=Compatibility%3A%20Linux%20%7C%20macOS%20%7C%20Windows&style=flat-square" alt="Compatibility: Linux | macOS | Windows"/>

</a>


## Overview


The repository is built around a main LaTeX document, with content split into chapter files under the sources directory. It also includes supporting assets such as bibliography entries, figures, and Python scripts used for analysis or plotting.


## Reproducibility and Generating Results

This repository also contains the configuration files, post-processing scripts, and raw telemetry data required to generate the figures and analyses for the seminar report. 

> **Note:** The full simulation execution framework (the SynxFlow solver and Alumet orchestration) is maintained in the [main project repository](https://github.com/thealanjason/topographic_uq_energy). To save setup time, this report repository is structured strictly for document compilation and data post-processing.

---

### 1. Generating the Report Figures

To recreate the plots using the provided dataset, follow these steps:

#### Step 1: Unpack the Data Archives

Extract `.tar.gz` archives inside the `data_archive/` directory using the terminal:

```bash
tar -xzvf data_archive/time_proxy_test.tar.gz -C data_archive/
tar -xzvf data_archive/uq_energy.tar.gz -C data_archive/
tar -xzvf data_archive/uq_energy_combined.tar.gz -C data_archive/
tar -xzvf data_archive/uq_energy_10_july_2026.tar.gz -C data_archive/
```

#### Step 2: Set Up the Environment

Create the `env-energy-analysis` environment using Micromamba and the provided `environment.yml` file:

```bash
micromamba env create -f environment.yml -n env-energy-analysis
```

#### Step 3: Run the Analysis Script

Execute the main post-processing and plotting script via Micromamba:

```bash
micromamba run -n env-energy-analysis python scripts/analyze_energy_proxy.py
```

Once execution completes, all generated figures will be saved in the `plots/` directory.

---

### 2. Recreating the Simulation Experiments

To execute full Monte Carlo ensembles and generate new telemetry data from scratch:

1. Clone the [main project repository](https://github.com/thealanjason/topographic_uq_energy) containing the live SynxFlow and Alumet integration.
2. Copy the configuration setups from `configs/` and the execution scripts from `scripts/` in this repository.
3. Replace the corresponding files in the main project repository's structure.
4. Run the simulation pipeline within the main project environment.
