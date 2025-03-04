import time
from zmqRemoteApi import RemoteAPIClient

client = RemoteAPIClient()
sim = client.getObject('sim')

client.setStepping(True)

sim.startSimulation()
while (t := sim.getSimulationTime()) < 3:
    s = f'Simulation time: {t:.2f} [s]'
    h = sim.getObject('/Floor')
    print(h)
    print(s)
    print("-----")
    client.step()
sim.stopSimulation()