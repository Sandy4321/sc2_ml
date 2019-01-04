from building import terran_buildings, protoss_buildings, zerg_buildings
from training import terran_fighting_units, protoss_fighting_units, zerg_fighting_units
from training import terran_units, protoss_units, zerg_units
from parsing import Parse
import numpy as np
import pandas as pd
from collections import deque, Counter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from operator import itemgetter
from spawningtool import parser
from glob import glob

units = protoss_fighting_units+zerg_fighting_units+terran_fighting_units
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

class UnitOrders():
    """
    Extract produced units from `.sc2replay` files.
    """
    def __init__(self, datasetfiles):
        self.files = datasetfiles
        self.df = pd.DataFrame()

    def parse(self, best_counters=False):
        """
        Use the spawningtool parser for `.sc2replay` file.
        :param file:
        :return: replay
        """
        prod = []
        max_prod = []
        counter = []
        n_counter = []
        for file in self.files:
            try:
                replay = parser.parse_replay(file)
                if best_counters:
                    p, mp, c, nc = self.most_produced_counter(replay)
                    prod.append(p)
                    max_prod.append(mp)
                    counter.append(c)
                    n_counter.append(nc)
                else:
                    df1 = self.extract_units(replay, 1)
                    df2 = self.extract_units(replay, 2)
                    self.df = pd.concat([self.df, df1, df2], sort=True)
            except:
                pass
        if best_counters:
            self.df = pd.DataFrame({'most_produced':prod, 'n_produced': max_prod, 'best_counter': counter, 'counter_prod': n_counter})
        else:
            self.df.reset_index(inplace=True, drop=True)
        return self.df

    def bo(self, replay, player):
        """
        Extract the number of produced units.
        :param replay: parsed replay file
        :param player: int
        :return: Counter dict
        """
        return Counter([x['name'] for x in replay['players'][player]['buildOrder'] if x['name'] in units])

    def loss(self, replay, player):
        """
        Extract the number of killed units.
        :param replay: parsed replay file
        :param player: int
        :return: Counter dict
        """
        return Counter([x['name'] for x in replay['players'][player]['unitsLost'] if x['name'] in units])

    def survival(self, produced, loss):
        """
        Compute the ratio of surviving units
        :param produced: dict of produced units
        :param loss: dict of lost units
        :return: dict of survival ratios
        """
        survival = {}
        for k, v in produced.items():
            if k in loss:
                survival[k] = (v - loss[k]) / v
            else:
                survival[k] = 1.0
        return survival

    def to_df(self, d, prefix):
        """
        Convert dict into dataframe with `prefix`ed column names
        :param d: dict
        :param prefix: str
        :return: pandas df
        """
        return pd.DataFrame.from_dict(d, orient='index').T.add_prefix(prefix)

    def extract_units(self, replay, player):
        """
        Extract unit statistics as dataframe.
        :param replay: parsed replay file
        :param player: int
        :return: pandas df
        """
        l = [2, 1, 2, 1]
        opponent = l[player + 1]
        player_produced = self.bo(replay, player)
        player_loss = self.loss(replay, player)
        player_survival = self.survival(player_produced, player_loss)
        opponent_produced = self.bo(replay, opponent)
        opponent_loss = self.loss(replay, opponent)
        opponent_survival = self.survival(opponent_produced, opponent_loss)
        win = pd.DataFrame([replay['players'][player]['is_winner']], columns=['is_winner'])
        d1 = self.to_df(player_produced, 'prod_')
        d2 = self.to_df(player_loss, 'loss_')
        d3 = self.to_df(player_survival, 'surv_')
        d4 = self.to_df(opponent_produced, 'oppo_prod_')
        d5 = self.to_df(opponent_loss, 'oppo_loss_')
        d6 = self.to_df(opponent_survival, 'oppo_surv_')
        return pd.concat([d1, d2, d3, d4, d5, d6, win], axis=1)

    def most_produced_counter(self, replay):
        player = [x for x in [1,2] if not replay['players'][x]['is_winner']][0]
        opponent = [x for x in [1,2] if replay['players'][x]['is_winner']][0]

        player_produced = self.bo(replay, player)
        max_prod = player_produced.most_common(1)[0]

        opponent_produced = self.bo(replay, opponent)
        opponent_loss = self.loss(replay, opponent)
        opponent_survival = self.survival(opponent_produced, opponent_loss)
        counter = Counter(opponent_survival).most_common(1)[0][0]
        prod_counter = opponent_produced[counter]

        return max_prod[0], max_prod[1], counter, prod_counter