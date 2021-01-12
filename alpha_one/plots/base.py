from alpha_one.utils.io import create_directories
from env import PLOTS_DIR
import matplotlib.pyplot as plt


class PlotManager:

    def __init__(self, game_name: str, run_name: str):
        self.game_name = game_name
        self.run_name = run_name

    def save_current_plot(self, plot_name: str):
        plot_name = plot_name.lower().strip().replace(' ', '_')
        file_path = f"{PLOTS_DIR}/{self.game_name}/{self.run_name}/{plot_name}"
        create_directories(file_path)
        plt.savefig(file_path)
