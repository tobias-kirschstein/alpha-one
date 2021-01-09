from envyaml import EnvYAML

env = EnvYAML('env.yaml')

# Add all variables from env.yaml here that you intend to directly import somewhere else

MODEL_SAVES_DIR = env['model_saves_dir']
LOGS_DIR = env['logs_dir']
PLOTS_DIR = env['plots_dir']