"""Thesis build helper: VS Code setup and TinyTeX package management."""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

THESIS = "thesis.tex"
PACKAGES_FILE = Path(__file__).parent / "tex-packages.txt"


def _ensure_system_path():
    if sys.platform == "win32":
        return
    system_bins = ["/usr/local/bin", "/usr/bin", "/bin"]
    current = os.environ.get("PATH", "").split(os.pathsep)
    additions = [p for p in system_bins if p not in current]
    if additions:
        os.environ["PATH"] = os.pathsep.join(current + additions)


def _read_packages():
    if not PACKAGES_FILE.exists():
        return []
    return [p for p in PACKAGES_FILE.read_text().splitlines() if p.strip()]


def _write_vscode_tool(tool):
    """Upsert a single tool entry in .vscode/settings.json by name."""
    vscode_dir = Path(__file__).parent / ".vscode"
    vscode_dir.mkdir(exist_ok=True)
    settings_file = vscode_dir / "settings.json"
    settings = {}
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text())
        except json.JSONDecodeError:
            pass
    settings.pop("latex-workshop.latex.recipes", None)
    tools = [t for t in settings.get("latex-workshop.latex.tools", []) if t["name"] != tool["name"]]
    tools.append(tool)
    settings["latex-workshop.latex.tools"] = tools
    settings_file.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Configured '{tool['name']}' in .vscode/settings.json")


def cmd_setup(args):
    if not args.tectonic and not args.tinytex:
        sys.exit("Specify --tectonic, --tinytex, or both.")

    _ensure_system_path()

    if args.tectonic:
        tectonic = shutil.which("tectonic")
        if not tectonic:
            sys.exit("Error: tectonic not found. Is the conda environment activated?")
        _write_vscode_tool({
            "name": "tectonic",
            "command": tectonic,
            "args": ["-X", "compile", "%DOC%.tex"],
            "env": {},
        })

    if args.tinytex:
        import pytinytex

        prefix = os.environ.get("CONDA_PREFIX")
        if not prefix:
            sys.exit("Error: activate your conda/micromamba environment first.")

        target = os.path.join(prefix, "share", "tinytex")
        try:
            bin_path = pytinytex.get_tinytex_path(target)
            print("TinyTeX already installed, skipping download.")
        except RuntimeError:
            with tempfile.TemporaryDirectory() as tmpdir:
                pytinytex.download_tinytex(variation=2, target_folder=target, download_folder=tmpdir)
            bin_path = pytinytex.get_tinytex_path(target)

        activate_d = os.path.join(prefix, "etc", "conda", "activate.d")
        deactivate_d = os.path.join(prefix, "etc", "conda", "deactivate.d")
        os.makedirs(activate_d, exist_ok=True)
        os.makedirs(deactivate_d, exist_ok=True)

        system_paths = "/usr/local/bin:/usr/bin:/bin"
        with open(os.path.join(activate_d, "tinytex.sh"), "w") as f:
            f.write('export _OLD_TINYTEX_PATH="$PATH"\n')
            f.write(f'export PYTINYTEX_TINYTEX="{target}"\n')
            f.write(f'export PATH="{bin_path}:${{PATH:+$PATH:}}{system_paths}"\n')
        with open(os.path.join(deactivate_d, "tinytex.sh"), "w") as f:
            f.write('export PATH="$_OLD_TINYTEX_PATH"\n')
            f.write('unset _OLD_TINYTEX_PATH\n')
            f.write('unset PYTINYTEX_TINYTEX\n')
        with open(os.path.join(activate_d, "tinytex.bat"), "w") as f:
            f.write('@SET "_OLD_TINYTEX_PATH=%PATH%"\n')
            f.write(f'@SET "PYTINYTEX_TINYTEX={target}"\n')
            f.write(f'@SET "PATH={bin_path};%PATH%"\n')
        with open(os.path.join(deactivate_d, "tinytex.bat"), "w") as f:
            f.write('@SET "PATH=%_OLD_TINYTEX_PATH%"\n')
            f.write('@SET "_OLD_TINYTEX_PATH="\n')
            f.write('@SET "PYTINYTEX_TINYTEX="\n')

        packages = _read_packages()
        if packages:
            print(f"\nInstalling {len(packages)} LaTeX package(s) from {PACKAGES_FILE.name}...")
            for pkg in packages:
                pytinytex.install(pkg)

        full_path = f"{bin_path}:{os.environ.get('PATH', '')}"
        _write_vscode_tool({
            "name": "latexmk",
            "command": "latexmk",
            "args": ["-synctex=1", "-interaction=nonstopmode", "-file-line-error", "%DOC%"],
            "env": {"PATH": full_path},
        })
        print(f"TinyTeX installed to: {bin_path}")


def cmd_build(args):
    import subprocess
    _ensure_system_path()
    import pytinytex

    while True:
        result = pytinytex.compile(THESIS, engine="xelatex", auto_install=True)
        if result.installed_packages:
            existing = _read_packages()
            for pkg in result.installed_packages:
                if pkg not in existing:
                    existing.append(pkg)
                    with PACKAGES_FILE.open("a") as f:
                        f.write(pkg + "\n")
                    print(f"Added '{pkg}' to {PACKAGES_FILE.name}")
        if result.success:
            break
        if not result.installed_packages:
            for entry in result.errors:
                print(f"Error: {entry.message}")
            sys.exit("Build failed — no new packages found. Check thesis.log.")

    print("\nAll packages installed — running full build...")
    sys.exit(subprocess.run(["latexmk", THESIS]).returncode)


def cmd_install(args):
    _ensure_system_path()
    import pytinytex

    existing = _read_packages()
    for pkg in args.packages:
        pytinytex.install(pkg)
        if pkg not in existing:
            existing.append(pkg)
            with PACKAGES_FILE.open("a") as f:
                f.write(pkg + "\n")
            print(f"Added '{pkg}' to {PACKAGES_FILE.name}")


def main():
    parser = argparse.ArgumentParser(description="Thesis build helper.")
    sub = parser.add_subparsers(dest="command")

    p_setup = sub.add_parser("setup", help="Configure LaTeX Workshop for the selected tool(s)")
    p_setup.add_argument("--tectonic", action="store_true", help="Configure Tectonic")
    p_setup.add_argument("--tinytex", action="store_true", help="Install TinyTeX and configure latexmk")

    sub.add_parser("build", help="Auto-install missing LaTeX packages then build with latexmk")

    p_install = sub.add_parser("install", help="Install LaTeX packages and save them to tex-packages.txt")
    p_install.add_argument("packages", nargs="+", help="Package name(s)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    {"setup": cmd_setup, "build": cmd_build, "install": cmd_install}[args.command](args)


if __name__ == "__main__":
    main()
