
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from Constant import Constant

import constants_ui

class BasketballAnimation():
    def __init__(self, mbd, x, id_team, id_player, t):
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = plt.axes(xlim=(Constant.X_MIN, Constant.X_MAX-Constant.DIFF),
                           ylim=(Constant.Y_MIN, Constant.Y_MAX))
        self.x, self.id_team, self.id_player = x, id_team, id_player
        self.mbd = mbd
        # self.ax.axis('off')
        self.ax.grid(True, 'both')
        n_players = x.shape[-2]
        
        self.clock_info = self.ax.annotate('', xy=[Constant.X_CENTER, Constant.Y_CENTER],
                                 color='black', horizontalalignment='center',
                                 verticalalignment='center')

        self.circle_texts = [self.ax.annotate('', xy=[0, 0], color='w',
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold')
                       for player in range(n_players)]

        # Prepare table
        # sorted_players = sorted(start_moment.players, key=lambda player: player.team.id)
        team_ids = id_team[0, [1,-1]].tolist()
        
        # home_player = sorted_players[0]
        # guest_player = sorted_players[5]
        column_labels = [mbd.team_id2data[i]['abbreviation'] for i in team_ids]
        column_colours = [constants_ui.team_id2color[i] for i in team_ids]
        cell_colours = [column_colours for _ in range(5)]
        
        players_home = [f"{mbd.player_id2data[i]['firstname']} {mbd.player_id2data[i]['lastname']} "+
                        f"#{mbd.player_id2data[i]['jersey']} {mbd.player_id2data[i]['position']}" for i in id_player[0][1:6].tolist()]
        players_away = [f"{mbd.player_id2data[i]['firstname']} {mbd.player_id2data[i]['lastname']} "+
                        f"#{mbd.player_id2data[i]['jersey']} {mbd.player_id2data[i]['position']}" for i in id_player[0][6:11].tolist()]
        # players_home = [f"{mbd.player_id2data[i]['firstname']} {mbd.player_id2data[i]['lastname']} " for i in ds['id_player'][0][1:6].tolist()]
        # players_away = [f"{mbd.player_id2data[i]['firstname']} {mbd.player_id2data[i]['lastname']} " for i in ds['id_player'][0][6:11].tolist()]
        players_data = list(zip(players_home, players_away))
        table = plt.table(cellText=players_data,
                          colLabels=column_labels,
                          colColours=column_colours,
                          colWidths=[.5]*2,
                          loc='bottom',
                          cellColours=cell_colours,
                          fontsize=Constant.FONTSIZE,
                          cellLoc='center')
        table.scale(1, Constant.SCALE)
        for cell in table.get_celld().values():
            cell.set_text_props(c='white')

        self.circles = [plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color='r')
                          for player in range(n_players)]
        for circle in self.circles:
            self.ax.add_patch(circle)

        self.anim = FuncAnimation(
                         self.fig, self.animation_step,
                         init_func=self.animation_init,
                         # fargs=(self.circles,),
                         frames=len(x), interval=20, blit=True,
                         repeat=True)
        img_court = plt.imread("court.png")
        plt.imshow(img_court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF, Constant.Y_MAX, Constant.Y_MIN])
        
    def animation_init(self):
        return self.animation_step(0)
    
    def animation_step(self, timestep):
        for circle_i, circle in enumerate(self.circles):
            circle.center = tuple(self.x[timestep, circle_i, :2].numpy())
            
        id_team = self.id_team[timestep]
        id_player = self.id_player[timestep]
        for i_player, (circle, circle_text) in enumerate(zip(self.circles, self.circle_texts)):
            circle.set_color(constants_ui.team_id2color[id_team[i_player].item()])
            circle_text.set_position(circle.center)
            player_data = self.mbd.player_id2data[id_player[i_player].item()]
            if 'jersey' in player_data:
                circle_text.set_text(f"{player_data['jersey']}{player_data['position']}")
            
        return self.circles#+self.circle_texts
    
    # def update_radius(self, i, circles, ball_circle, annotations, clock_info):
    #     moment = self.moments[i]
    #     for j, circle in enumerate(circles):
    #         circle.center = moment.players[j].x, moment.players[j].y
    #         annotations[j].set_position(circle.center)
    #         clock_test = 'Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}'.format(
    #                      moment.quarter,
    #                      int(moment.game_clock) % 3600 // 60,
    #                      int(moment.game_clock) % 60,
    #                      moment.shot_clock)
    #         clock_info.set_text(clock_test)
    #     ball_circle.center = moment.ball.x, moment.ball.y
    #     ball_circle.radius = moment.ball.radius / Constant.NORMALIZATION_COEF
    #     return circles, ball_circle
