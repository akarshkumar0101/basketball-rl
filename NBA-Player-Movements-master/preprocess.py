import numpy as np

def load_data_game(game, tqdm=None):
    data_game = []
    pbar = game.events
    if tqdm is not None:
        pbar = tqdm(pbar)
    for i_event, event in enumerate(pbar):
        if len(event.moments)==0:
            continue
        data_e = np.full((len(event.moments), 5+3+40), fill_value=np.nan)
        for i_moment, moment in enumerate(event.moments):
            ball = moment.ball
            quarter_clock = moment.game_clock
            game_clock = quarter_clock+(4-moment.quarter)*12*60
            # game_clock = int((game_clock+.005)*100)/100.
            
            data_e[i_moment, :5] = [game_clock, quarter_clock, moment.shot_clock, moment.quarter, 0]#i_event]
            # data_m.extend([moment.game_clock, moment.shot_clock, moment.quarter])
            data_e[i_moment, 5:8] = [ball.x, ball.y, ball.radius]
            # data_m.extend([ball.x, ball.y, ball.radius])

            for i_player, player in enumerate(moment.players):
                data_e[i_moment, 8+(i_player)*4: 8+(i_player+1)*4] = [player.id, player.team.id, player.x, player.y]
                # data_m.extend([player.id, player.team.id, player.x, player.y])
                # sometimes there is not enough players in this loop
                # so manually create a array of nans at the beginning (rather than at the end)
                # and fill the correct player positions
            # data_e.append(data_m)

        # print(np.array([len(i) for i in data_e]).std())
        # print(np.array([len(i) for i in data_e]).min())
        # print(np.array([len(i) for i in data_e]).max())
        # print()
        # data_e = np.array(data_e)
        
        # only add if this new event is different from the previous one
        
        if len(data_game)==0 or not np.array_equal(data_e, data_game[-1], equal_nan=True):
            data_game.append(data_e)
    return data_game
    
def preprocess(data_game):
    # concatenate all events' moments
    data_game = np.concatenate(data_game, axis=0).T
    t = data_game[0]
    
    # only include the first instance of a time stamp (to two decimals)
    t_binned = ((t+0.001)*100).astype(int)
    _, idxs = np.unique(t_binned, return_index=True)
    data_game = data_game[:, idxs]
    
    # sort the times in case they're out of order
    idxs = np.argsort(data_game[0])[::-1]
    data_game = data_game[:, idxs]
    return data_game.T
        
        # break
    # break
# ts, bx = np.array(ts), np.array(bx)
# idxs = []
# t_seen = set()
# pt = 720
# for i, t in enumerate(((ts+0.0001)*100).astype(int)):
#     if pt<5 and t>600:
#         t_seen = set()
#     if t not in t_seen:
#         t_seen.add(t)
#         idxs.append(i)
#     pt = t


# i = np.array(idxs)
# ts, bx = ts[i], bx[i]