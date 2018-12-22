import random
from parsing import Parse
import pandas as pd
import numpy as np

terran_units = [
    'SCV',
    'MULE',
    'Marine',
    'Marauder',
    'Reaper',
    'Ghost',
    'Hellion',
    'Hellbat',
    'SiegeTank',
    'Cyclone',
    'WidowMine',
    'Thor',
    'Viking',
    'Medivac',
    'Liberator',
    'Raven',
    'Banshee',
    'Battlecruiser'
]

zerg_units = [
    'Larva',
    'Drone',
    'Queen',
    'Zergling',
    'Baneling',
    'Roach',
    'Ravager',
    'Hydralisk',
    'Lurker',
    'Infestor',
    'SwarmHost',
    'Ultralisk',
    'Locust',
    'Broodling',
    'Changeling',
    'InfestedTerran',
    'NydusWorm',
    'Overlord',
    'Lair',
    'Overseer',
    'Mutalisk',
    'Corruptor',
    'Hive',
    'BroodLord',
    'Viper'
]

protoss_units = [
    'Probe',
    'Zealot',
    'Stalker',
    'Sentry',
    'Adept',
    'HighTemplar',
    'DarkTemplar',
    'Immortal',
    'Colossus',
    'Disruptor',
    'Archon',
    'Observer',
    'WarpPrism',
    'Phoenix',
    'VoidRay',
    'Oracle',
    'Carrier',
    'Tempest',
    'MothershipCore',
    'Mothership'
]

class TrainingOrders():
    def __init__(self, race, datasetfiles, nb_to=1000):
        self.race = race
        self.datafiles = random.sample(datasetfiles, nb_to)
        if self.race == 'Zerg':
            self.training_list = zerg_units
        elif self.race == 'Protoss':
            self.training_list = protoss_units
        elif self.race == 'Terran':
            self.training_list = terran_units
        else:
            self.training_list = None
            raise ValueError('Unknown race %s' % self.race)

    def select_to(self, winners=True):
        tos = []
        for dataset in self.datafiles:
            sc = Parse(dataset)
            if winners:
                if sc.p1_race == self.race and sc.winner == 0:
                    tos.append(sc.p1_train)
                if sc.p2_race == self.race and sc.winner == 1:
                    tos.append(sc.p2_train)
            else:
                if sc.p1_race == self.race and sc.winner == 1:
                    tos.append(sc.p1_train)
                if sc.p2_race == self.race and sc.winner == 0:
                    tos.append(sc.p2_train)
        return tos

    def train_orders(self, max_steps=15000, winners=True):
        tos = self.select_to()
        mat = np.zeros((len(tos), len(self.training_list), max_steps))
        for i,t in enumerate(tos):
            for row in t:
                if (int(row[0]) < max_steps) and (row[1] in self.training_list):
                    mat[i, int(self.training_list.index(row[1])), int(row[0])] = 1.
        frames_per_sec = int(1000/11.278)
        n_steps = int(max_steps/frames_per_sec)
        s = zip(np.arange(0, (n_steps-1)*frames_per_sec, frames_per_sec),
                np.arange(frames_per_sec, n_steps*frames_per_sec, frames_per_sec))
        mat_per_sec = np.array([mat[:,:,x:y] for x,y in s])
        count_mat_per_sec = np.sum(mat_per_sec, axis=3)
        mat_mean = np.mean(count_mat_per_sec, axis=1)
        return pd.DataFrame(mat_mean, columns=self.training_list, index=pd.timedelta_range(0, periods=n_steps-1, freq='S'))

