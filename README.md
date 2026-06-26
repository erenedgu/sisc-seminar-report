# Thesis Template

A LaTeX thesis template based on [latex-mimosis](https://github.com/Pseudomanifold/latex-mimosis), with a basic chapter structure and student guidance built in.

<a href="https://raw.githubusercontent.com/thealanjason/thesis-template/pdf-tectonic/thesis.pdf">
<img src="https://img.shields.io/badge/View-PDF_(Tectonic)-red?style=flat-square&logo=adobeacrobatreader&logoColor=white" alt="View the thesis PDF (Tectonic)"/>
</a>
<a href="https://raw.githubusercontent.com/thealanjason/thesis-template/pdf-tinytex/thesis.pdf">
<img src="https://img.shields.io/badge/View-PDF_(TinyTeX)-red?style=flat-square&logo=adobeacrobatreader&logoColor=white" alt="View the thesis PDF (TinyTeX)"/>
</a>
<a href="https://github.com/thealanjason/thesis-template/actions/workflows/compat.yml">
<img src="https://img.shields.io/github/actions/workflow/status/thealanjason/thesis-template/compat.yml?label=Compatibility%3A%20Linux%20%7C%20macOS%20%7C%20Windows&style=flat-square" alt="Compatibility: Linux | macOS | Windows"/>
</a>

## Structure

```
thesis.tex                  # Main file — configure title, author, and thesis type here
sources/
  title.tex                 # Title page
  abstract.tex              # Abstract
  acknowledgements.tex      # Acknowledgements
  declarationofownwork.tex  # Declaration of own work + AI tool usage
  conventions.tex           # Notation and writing conventions
  introduction.tex          # Chapter 1
  relatedwork.tex           # Chapter 2
  ownwork.tex               # Chapter 3 — your contribution
  evaluation.tex            # Chapter 4
  summaryandfuturework.tex  # Chapter 5
  appendix.tex              # Appendix
resources/
  logoipsum-354.pdf         # Replace with your institution's logo
  declarationofacademicintegrity.pdf  # Required for Master's and Bachelor's theses at RWTH Aachen University
images/                     # Place your figures here
data/                       # Place your datasets here
scripts/                    # Python scripts to generate plots, with their conda environments
thesis.bib                  # Bibliography
```

## Getting started

1. Click **Use this template** on GitHub to create your own repository
2. Clone it locally:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
3. Create and activate the environment:
   ```bash
   micromamba env create -f environment.yml
   micromamba activate thesis
   ```
4. Build the thesis to verify everything works (see [Building](#building) below)
5. Then make it yours:
   - Set your title, author, and thesis type in `thesis.tex`:
     ```latex
     \masterthesistrue  % or \bachelorthesistrue — leave both false for project reports
     \title{Your Thesis Title}
     \subtitle{Your Subtitle}
     \author{Your Name}
     ```
   - Replace `resources/logoipsum-354.pdf` with your institution's logo
   - Update the badge URLs in this README (replace `thealanjason/thesis-template` with your username/repo)

## Building

### Option A — Tectonic

No TeX Live installation needed — packages are auto-downloaded on first run.

**Command line:**
```bash
tectonic -X compile thesis.tex
```

**LaTeX Workshop:** Install the [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) extension, run once to configure:
```bash
python thesis.py setup --tectonic
```
Then open `thesis.tex`, select **Recipe: tectonic** and click **Build**, then **View LaTeX PDF**.

### Option B — latexmk (TinyTeX)

Full TeX Live toolchain. Install TinyTeX once, then build:

**Command line:**
```bash
python thesis.py setup --tinytex
python thesis.py build
```
`build` auto-installs any missing LaTeX packages. For subsequent builds:
```bash
latexmk
```

**LaTeX Workshop:** Install the [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) extension, run once to configure:
```bash
python thesis.py setup --tinytex
```
Then open `thesis.tex`, select **Recipe: latexmk** and click **Build**, then **View LaTeX PDF**.

---

The thesis is built automatically on every push via GitHub Actions using both options.
