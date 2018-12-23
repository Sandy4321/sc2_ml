from building import terran_buildings, protoss_buildings, zerg_buildings
from training import terran_units, protoss_units, zerg_units
from parsing import Parse

#ToDO: Define Orders class with select and orders methods

#ToDo: Define a GameEvents class to produce the build and train matrix of frames events

class Orders():
    def __init__(self, race, eventtype, datasetfiles):
        self.race = race
        self.eventtype = eventtype
        self.object_list = self.choose(race, eventtype)

    def choose(self, race, eventtype):
        x = race.lower() + '_' + eventtype.lower()
        return {
            'terran_units': terran_units,
            'zerg_units': zerg_units,
            'protoss_units': protoss_units,
            'terran_buildings': terran_buildings,
            'zerg_buildings': zerg_buildings,
            'protoss_buildings': protoss_buildings
        }.get(x, None)

    def select(self, datasetfiles, number_of_games=10):
        orders_list = []
        winner_list = []
        f = iter(datasetfiles)
        while len(orders_list) < number_of_games:
            data = next(f, None)
            if data:
                sc = Parse(data)
                if sc.p1_race == self.race:
                    orders_list.append(sc.extract(1, self.eventtype))
                    winner_list.append(sc.winner == 0)
                elif sc.p2_race == self.race:
                    orders_list.append(sc.extract(2, self.eventtype))
                    winner_list.append(sc.winner == 1)
        return orders_list, winner_list

    def orders(self, max_frames):
        pass

