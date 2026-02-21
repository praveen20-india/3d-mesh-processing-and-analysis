#!/bin/bash
cp ../full_dataset_mesh.stl data/
cp ../half_dataset_mesh.stl data/
cp ../third_dataset_mesh.stl data/
cp ../quarter_dataset_mesh.stl data/

source venv/bin/activate
python3 src/build_notebook.py
python3 -m src.main --save-plots
