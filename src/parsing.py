import pandas as pd
import re

class Parse():
    def __init__(self, datasetfile):
        self.data = self.extract_data(datasetfile)
        self.players_names = []
        self.p1_race = None
        self.p2_race = None
        try:
            self.players_names = self.extract_players_name(self.data)
            self.p1, self.p2 = self.split_players(self.data)
            self.p1_train = self.extract_training(1)
            self.p2_train = self.extract_training(2)
            self.winner = self.get_winner()
            self.p1_race, self.p2_race = self.extract_races()
        except IndexError:
            # print('Not a dual game.')
            pass


    def extract_players_name(self, data):
        return list(set([x.split('\t')[1] for x in data]))

    def extract_data(self, datasetfile):
        with open(datasetfile, 'r') as file:
            return file.readlines()[2:]

    def split_players(self, data):
        player1 = []
        player2 = []
        for line in data:
            columns = line.split('\t')
            if columns[1] == self.players_names[0]:
                player1.append(columns[::2])
            elif columns[1] == self.players_names[1]:
                player2.append(columns[::2])
        return player1, player2

    def extract_positions(self, players_moves):
        frames = []
        x_positions = []
        y_positions = []
        for line in players_moves:
            if re.match(r'Move screen', line[1]):
                m = re.search(r'x=(\d+\.\d+),y=(\d+\.\d+)', line[1])
                frames.append(int(line[0]))
                x_positions.append(float(m.group(1)))
                y_positions.append(float(m.group(2)))
        return pd.DataFrame({'frames': frames, 'x':x_positions, 'y':y_positions})

    def extract_move_screen(self, as_density_matrix=False):
        p1_moves = self.extract_positions(self.p1)
        p2_moves = self.extract_positions(self.p2)
        p1_moves['player'] = 'p1'
        p2_moves['player'] = 'p2'
        if as_density_matrix:
            p1_moves['count'] = 1
            p1_moves.x = p1_moves.x.astype(int)
            p1_moves.y = p1_moves.y.astype(int)
            p2_moves['count'] = 1
            p2_moves.x = p2_moves.x.astype(int)
            p2_moves.y = p2_moves.y.astype(int)
            pt1 = pd.pivot_table(p1_moves, values='count', index='y', columns='x', aggfunc='sum')
            pt2 = pd.pivot_table(p2_moves, values='count', index='y', columns='x', aggfunc='sum')
            return pt1, pt2
        else:
            return p1_moves, p2_moves

    def extract_building(self, player):
        player_building = []
        if player == 1:
            data = self.p1
        elif player == 2:
            data = self.p2
        else:
            raise ValueError('Unknown player number: %d' % player)

        for line in data:
            frame = line[0]
            action_line = line[1].split(';')
            action_len = len(action_line)
            if re.match(r'Build', line[1]):
                t = re.search(r'(\w+)\(?(\w+)?\)?', ''.join(action_line[0].split(' ')[1:]))
                if t:
                    object_target = t.group(2)
                    build_type = t.group(1)
                m = re.search(r'x=(\d+\.\d+),y=(\d+\.\d+)', line[1])
                if m:
                    x_pos = float(m.group(1))
                    y_pos = float(m.group(2))
                else:
                    x_pos = None
                    y_pos = None

                player_building.append((frame, build_type, x_pos, y_pos, object_target))
        return player_building

    def get_winner(self):
        win = None
        if len(self.players_names) == 2:
            if self.p1[-1][1].strip() == 'Leave game':
                win = 0
            elif self.p2[-1][1].strip() == 'Leave game':
                win = 1
            else:
                win = 2
        return win

    def extract_training(self, player):
        player_training = []
        if player == 1:
            data = self.p1
        elif player == 2:
            data = self.p2
        else:
            raise ValueError('Unknown player number: %d' % player)

        for line in data:
            frame = line[0]
            action_line = line[1].split(';')
            if re.match(r'Train', line[1]):
                t = re.search(r'(\w+)', ''.join(action_line[0].split(' ')[1:]))
                if t:
                    player_training.append((frame, t.group(1)))
        return player_training

    def extract_races(self):
        if self.p1_train[0][1] == 'SCV':
            p1 = 'Terran'
        elif self.p1_train[0][1] == 'Drone':
            p1 = 'Zerg'
        elif self.p1_train[0][1] == 'Probe':
            p1 = 'Protoss'
        else:
            p1 = None
        if self.p2_train[0][1] == 'SCV':
            p2 = 'Terran'
        elif self.p2_train[0][1] == 'Drone':
            p2 = 'Zerg'
        elif self.p2_train[0][1] == 'Probe':
            p2 = 'Protoss'
        else:
            p2 = None
        return p1, p2

    def extract(self, player, eventtype):
        if eventtype == 'units':
            return self.extract_training(player)
        elif eventtype == 'training':
            return self.extract_building(player)
        else:
            raise ValueError('Not a valid option: %s' % eventtype)