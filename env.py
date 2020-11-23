from envyaml import EnvYAML

env = EnvYAML('env.yaml')

# Add all variables from env.yaml here that you intend to directly import somewhere else

EXAMPLE_VARIABLE = env['test']