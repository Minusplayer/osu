{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python312;
  pythonEnv = python.withPackages (ps: with ps; [
    # NOTE: torch intentionally NOT here — current nixpkgs python312Packages.torch
    # has a broken eval (sphinx-9.1.0 incompatible). Installed via pip in the
    # venv overlay below as a workaround. Revisit once nixpkgs is fixed.
    numpy
    matplotlib
    # input injection + dev
    evdev
    ipython
    jupyter
    # for the tiny venv overlay (osrparse not in nixpkgs)
    pip
    virtualenv
  ]);
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    pythonEnv

    # screen capture (deferred; only if we go pixel-input later)
    gpu-screen-recorder
    ffmpeg
    grim
    slurp

    # input/device debugging
    libevdev
    evtest
  ];

  # pip-installed torch wheel needs libstdc++ at runtime.
  # /run/opengl-driver/lib exposes the NixOS NVIDIA driver userspace
  # (libcuda.so etc.) so torch.cuda.is_available() returns True.
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ] + ":/run/opengl-driver/lib";

  shellHook = ''
    # venv overlay for packages not (currently) available from nixpkgs
    if [ ! -d .venv ]; then
      echo "creating venv overlay (osrparse + torch)..."
      ${pythonEnv}/bin/python -m venv --system-site-packages .venv
      .venv/bin/pip install --quiet osrparse torch
    fi
    source .venv/bin/activate
    echo ""
    echo "aiosu dev shell"
    echo "  python:   $(python --version)"
    echo "  torch:    $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'MISSING — rm -rf .venv and re-enter shell')"
    echo "  numpy:    $(python -c 'import numpy; print(numpy.__version__)')"
    echo "  osrparse: $(python -c 'import osrparse; print(osrparse.__version__)')"
    echo ""
  '';
}
