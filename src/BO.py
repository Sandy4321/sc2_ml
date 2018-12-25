from building import terran_buildings, protoss_buildings, zerg_buildings
from training import terran_units, protoss_units, zerg_units
from parsing import Parse
import numpy as np
import pandas as pd
from collections import deque
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from operator import itemgetter

FRAMES_PER_SEC = int(1000/11.278)

class Orders():
    def __init__(self, category):
        self.race, self.vs = self.get_cat(category)
        self.objects_list = self.get_objects(self.race)

    def get_objects(self, race):
        return {'terran': terran_units + terran_buildings,
                'zerg': zerg_units + zerg_buildings,
                'protoss': protoss_units + protoss_buildings}.get(race.lower(), None)

    def convert(self, s):
        return {'t':'Terran', 'z':'Zerg','p':'Protoss'}.get(s, None)

    def get_cat(self, category):
        first_player,_, second_player = [x for x in category.lower()]
        return self.convert(first_player), self.convert(second_player)

    def select(self, datasetfiles):
        orders_list = []
        winner_list = []

        for data in datasetfiles:

            sc = Parse(data)
            if (sc.p1_race == self.race) and (sc.p2_race == self.vs):
                units = sc.extract(1, 'units')
                builds = sc.extract(1, 'buildings')
                orders_list.append(units + builds)
                winner_list.append(sc.winner == 0)


            elif (sc.p2_race == self.race) and (sc.p1_race == self.vs):
                units = sc.extract(2, 'units')
                builds = sc.extract(2, 'buildings')
                orders_list.append(units + builds)
                winner_list.append(sc.winner == 1)

        return orders_list, winner_list

    def relative_events(self, orders_list, depth=10):

        events = []
        for e in orders_list:
            orders = {}
            for obj in self.objects_list:
                orders[obj] = deque([0.0] * depth, depth)
            for line in e:
                if line[1] in self.objects_list:
                    orders[line[1]].appendleft(int(line[0]))
            a = pd.DataFrame.from_dict(orders)
            events.append(a.values)

        dist_mat = np.zeros((len(events), len(events)))
        for i, a in enumerate(events):
            for j, b in enumerate(events):
                dist_mat[i, j] = np.mean(np.abs(a - b))
        return dist_mat

    def orders(self, order_list, max_frames=15000):
        mat = np.zeros((len(order_list), len(self.objects_list), max_frames))
        for i,o in enumerate(order_list):
            for row in o:
                if (int(row[0]) < max_frames) and (row[1] in self.objects_list):
                    mat[i, int(self.objects_list.index(row[1])), int(row[0])] = 1.
        n_steps = int(max_frames/FRAMES_PER_SEC)
        windows = zip(np.arange(0, (n_steps-1)*FRAMES_PER_SEC, FRAMES_PER_SEC),
                      np.arange(FRAMES_PER_SEC, n_steps*FRAMES_PER_SEC, FRAMES_PER_SEC))
        mat_per_sec = np.array([mat[:,:,x:y] for x,y in windows])
        count_mat_per_sec = np.sum(mat_per_sec, axis=3)
        mat_mean = np.mean(count_mat_per_sec, axis=1)
        return pd.DataFrame(mat_mean,
                            columns=self.objects_list,
                            index=pd.timedelta_range(0,
                                                     periods=n_steps-1,
                                                     freq='S'))

    def win_rate(self, db, winners):
        winners = np.array(winners)
        wr = {}
        clusters = np.unique(db.labels_)
        for cluster in clusters:
            wr[cluster] = np.mean(winners[np.where(db.labels_ == cluster)[0]])
        return wr

    def winning_BO(self, datasetfiles, depth=20, max_frames=15000, eps=0.5, min_sample=3):
        order_list, winners = self.select(datasetfiles)
        mat = self.relative_events(order_list, depth=depth)
        X = StandardScaler().fit_transform(mat)
        db = DBSCAN(eps=eps, min_samples=min_sample).fit(X)
        wr = self.win_rate(db, winners)
        best_cluster = max(wr.items(), key=itemgetter(1))[0]
        ol = np.array(order_list)[np.where(db.labels_ == best_cluster)].tolist()
        df = self.orders(ol, max_frames=max_frames)
        return df