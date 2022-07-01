import pandas as pd
from Event import Event
from Team import Team
from Constant import Constant

class Game:
    """A class for keeping info about the games"""
    def __init__(self, path_to_json):
        # self.events = None
        self.home_team = None
        self.guest_team = None
        self.events = None
        self.path_to_json = path_to_json
        self.player_ids_dict = None

    def read_json(self, tqdm=None):
        df = pd.read_json(self.path_to_json)
        events = df['events']
        self.home_team = Team(events[0]['home']['teamid'])
        self.guest_team = Team(events[0]['visitor']['teamid'])
        # print(Constant.MESSAGE + str(last_default_index))
        print('loading ', len(events))
        pbar = events
        if tqdm is not None:
            pbar = tqdm(pbar)
        self.events = [Event(event) for event in pbar]
        self.player_ids_dict = self.events[0].player_ids_dict
        
    def start(self):
        self.event.show()
