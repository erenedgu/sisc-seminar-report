# Thesis Template

A LaTeX thesis template based on [latex-mimosis](https://github.com/Pseudomanifold/latex-mimosis), with a basic chapter structure and student guidance built in.

<a href="https://raw.githubusercontent.com/erenedgu/sisc-seminar-report/pdf-tectonic/thesis.pdf">
<img src="https://img.shields.io/badge/View-Report_(Tectonic)-red?style=flat-square&logo=adobeacrobatreader&logoColor=white" alt="View the seminar report PDF (Tectonic)"/>
</a>
<a href="https://raw.githubusercontent.com/erenedgu/sisc-seminar-report/pdf-tinytex/thesis.pdf">
<img src="https://img.shields.io/badge/View-Report_(TinyTeX)-red?style=flat-square&logo=adobeacrobatreader&logoColor=white" alt="View the seminar report PDF (TinyTeX)"/>
</a>
<a href="https://github.com/erenedgu/sisc-seminar-report/actions/workflows/compat.yml">
<img src="https://img.shields.io/github/actions/workflow/status/erenedgu/sisc-seminar-report/compat.yml?label=Compatibility%3A%20Linux%20%7C%20macOS%20%7C%20Windows&style=flat-square" alt="Compatibility: Linux | macOS | Windows"/>
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
  mbd_logo.pdf         # Replace with your institution's logo
  declarationofacademicintegrity.pdf  # Required for Master's and Bachelor's theses at RWTH Aachen University
images/                     # Place your figures here
data/                       # Place your datasets here
scripts/                    # Python scripts to generate plots, with their conda environments
thesis.bib                  # Bibliography
```