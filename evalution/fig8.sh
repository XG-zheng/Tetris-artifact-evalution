#!/bin/bash

# Build ablation kernel
echo '-- Build ablation kernel(about 3min)...'
./script/build_ablation.sh


echo '-- Run ablation experiment(a)'
python3 ablation_timing.py

echo '-- Draw fig8(a), store in ./figure/fig8-a.pdf(png)'
python3 draw_ablation_a.py

echo '-- Run ablation experiment(b)'
./script/memory_metric.sh

echo '-- Draw fig8(b), store in ./figure/fig8-b.pdf(png)'
python3 draw_ablation_b.py



