from alpha_one.utils.io import create_directories, save_pickled, load_pickled
from alpha_one.utils.logging import generate_run_name
from env import PLOTS_DIR
import matplotlib.pyplot as plt


class PlotManager:

    def __init__(self, game_name: str, run_name: str):
        self.game_name = game_name
        self.run_name = run_name

    @staticmethod
    def new_run(plot_group: str, prefix: str = 'run'):
        run_name = generate_run_name(f"{PLOTS_DIR}/{plot_group}", prefix)
        return PlotManager(plot_group, run_name)

    def cd(self, sub_dir: str):
        return PlotManager(f"{self.game_name}/{self.run_name}/{sub_dir}", "")

    def save_current_plot(self, plot_name: str):
        plot_name = plot_name.lower().strip().replace(' ', '_')
        file_path = f"{PLOTS_DIR}/{self.game_name}/{self.run_name}/{plot_name}"
        create_directories(file_path)
        plt.savefig(file_path)

    def save(self, obj, file_name: str):
        save_pickled(obj, f"{PLOTS_DIR}/{self.game_name}/{self.run_name}/{file_name}")

    def load(self, file_name: str):
        return load_pickled(f"{PLOTS_DIR}/{self.game_name}/{self.run_name}/{file_name}")
