import random
from parsing import Parse
import pandas as pd
import numpy as np

terran_buildings = [
    'CommandCenter',
    'PlanetaryFortress',
    'OrbitalCommand',
    'SupplyDepot',
    'Refinery',
    'Barracks',
    'EngineeringBay',
    'Bunker',
    'SensorTower',
    'MissileTurret',
    'Factory',
    'GhostAcademy',
    'Starport',
    'Armory',
    'FusionCore',
    'TechLab',
    'Reactor',
    'AutoTurret'
]

zerg_buildings = [
    'Hatchery',
    'SpineCrawler',
    'SporeCrawler',
    'Extractor',
    'SpawningPool',
    'EvolutionChamber',
    'RoachWarren',
    'BanelingNest',
    'CreepTumor',
    'Lair',
    'HydraliskDen',
    'LurkerDen',
    'InfestationPit',
    'Spire',
    'NydusNetwork',
    'Hive',
    'GreaterSpire',
    'UltraliskCavern'
]

protoss_buildings = [
    'Nexus',
    'Pylon',
    'Assimilator',
    'Gateway',
    'Forge',
    'CyberneticsCore',
    'PhotonCannon',
    'RoboticsFacility',
    'WarpGate',
    'Stargate',
    'TwilightCouncil',
    'RoboticsBay',
    'FleetBeacon',
    'TemplarArchives',
    'DarkShrine'
]

class BuildOrders():
    def __init__(self, race, datasetfiles, nb_bo=1000):
        self.race = race
        self.datafiles = random.sample(datasetfiles, nb_bo)
        if self.race == 'Zerg':
            self.building_list = zerg_buildings
        elif self.race == 'Protoss':
            self.building_list = protoss_buildings
        elif self.race == 'Terran':
            self.building_list = terran_buildings
        else:
            self.building_list = None
            raise ValueError('Unknown race %s' % self.race)

    def select_bo(self, winners=True):
        bos = []
        for dataset in self.datafiles:
            sc = Parse(dataset)
            if winners:
                if sc.p1_race == self.race and sc.winner == 0:
                    bos.append(sc.extract_building(1))
                if sc.p2_race == self.race and sc.winner == 1:
                    bos.append(sc.extract_building(2))
            else:
                if sc.p1_race == self.race and sc.winner == 1:
                    bos.append(sc.extract_building(1))
                if sc.p2_race == self.race and sc.winner == 0:
                    bos.append(sc.extract_building(2))
        return bos

    def build_orders(self, max_steps=15000, winners=True):
        bos = self.select_bo(winners)
        mat = np.zeros((len(bos), len(self.building_list), max_steps))
        for i,b in enumerate(bos):
            for row in b:
                if (int(row[0]) < max_steps) and (row[1] in self.building_list):
                    mat[i, int(self.building_list.index(row[1])), int(row[0])] = 1.
        frames_per_sec = int(1000/11.278)
        n_steps = int(max_steps / frames_per_sec)
        # rolling indexes
        s = zip(np.arange(0, (n_steps-1)*frames_per_sec, frames_per_sec),
                np.arange(frames_per_sec, n_steps*frames_per_sec, frames_per_sec))

        mat_per_sec = np.array([mat[:,:,x:y] for x,y in s])
        max_mat_per_sec = np.max(mat_per_sec, axis=3)
        mat_mean = np.mean(max_mat_per_sec, axis=1)
        return pd.DataFrame(mat_mean, columns=self.building_list,
                            index=pd.timedelta_range(0, periods=n_steps-1, freq='S'))

