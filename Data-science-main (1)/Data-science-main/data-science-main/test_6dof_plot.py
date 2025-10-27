import sys
sys.path.insert(0, 'Data-science')

from main_6dof import MissileSimulation6DoF

sim = MissileSimulation6DoF("SCUD-B")
sim.run_simulation(45, 90, 500)
sim.plot_detailed_results()
