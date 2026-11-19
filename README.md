# Large-Scale Car Cruise-Control Simulations

This repository contains Python simulation scripts and an exploratory notebook modeling 100-vehicle mixed-traffic flow. It evaluates the specified `pcruise` proportional-derivative Adaptive Cruise Control (ACC) controller against human drivers modeled with the Intelligent Driver Model (IDM) and two alternative ACC tunings (aggressive and conservative).

## Repository Layout

```text
outputs/                              Generated plots, HTML visualizations, and CSV summaries
.gitignore                            Git ignore rules
IDM_sim.ipynb                         Interactive Jupyter notebook for IDM exploration
README.md                             Project documentation and usage guide
requirements.txt                      Python package dependencies (NumPy, Matplotlib)
traffic_fuel_efficiency_experiment.py Equilibrium-initialized matched-speed fuel experiment
traffic_ring_simulation.py            100-vehicle closed-loop ring-road simulation
traffic_straight_simulation.py        100-vehicle finite straight-road platoon simulation & animation
traffic_time_space_simulation.py      30-vehicle straight-road time-space shockwave diagram
