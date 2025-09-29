# https://huggingface.co/learn/deep-rl-course/en/unit8/hands-on-sf
# DISCLAIMER: I can't get this script to run on GPU due to "invalid resource handle" CUDA error.

# %%
# Import and setup
import functools
import multiprocessing as mp
import os

from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec


# Registers all the ViZDoom environments
def register_vizdoom_envs():
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)


# Sample Factory allows the registration of a custom Neural Network architecture
# See https://github.com/alex-petrenko/sample-factory/blob/master/sf_examples/vizdoom/doom/doom_model.py for more details
def register_vizdoom_models():
    from sample_factory.algo.utils.context import global_model_factory
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)


def register_vizdoom_components():
    register_vizdoom_envs()
    register_vizdoom_models()


# parse the command line args and create a config
def parse_vizdoom_cfg(argv=None, evaluation=False):
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    # parameters specific to Doom envs
    add_doom_env_args(parser)
    # override Doom default values for algo parameters
    doom_override_defaults(parser)
    # second parsing pass yields the final configuration
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


# %%
# Train the agent

# # The scenario we train on today is health gathering
# other scenarios include "doom_basic", "doom_two_colors_easy", "doom_dm", "doom_dwango5", "doom_my_way_home", "doom_deadly_corridor", "doom_defend_the_center", "doom_defend_the_line"
def train_the_agent(env: str):
    cfg = parse_vizdoom_cfg(
        argv=[
            f"--env={env}",
            "--num_workers=1",  # 8 CPU workers
            "--num_envs_per_worker=1",  # 4 envs per worker → 32 total CPU envs
            "--worker_num_splits=1",  # set to 1 as a temporary workaround
            "--device=gpu",              # main process on GPU
            # "--device=cpu",  # main process on CPU
            # "--async_rl=False",
            # "--train_for_env_steps=4000000",
            "--train_for_env_steps=40000",
        ]
    )

    status = run_rl(cfg)
    return status

# %%
# Watch the agent play the game
def enjoy_the_agent(env: str):
    from sample_factory.enjoy import enjoy

    cfg = parse_vizdoom_cfg(
        argv=[
            f"--env={env}",
            "--num_workers=1",
            "--save_video",
            "--no_render",
            "--max_num_episodes=10",
        ],
        evaluation=True,
    )
    status = enjoy(cfg)

# %%
# Visualize the performance of the agent
def visualize_play():
    from base64 import b64encode
    from IPython.display import HTML

    mp4 = open(os.path.dirname(__file__) + "/train_dir/default_experiment/replay.mp4", "rb").read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    HTML(
        """
    <video width=640 controls>
        <source src="%s" type="video/mp4">
    </video>
    """
        % data_url
    )


# %%
if __name__ == "__main__":
    # Set the start method to 'forkserver' or 'spawn'
    # This must be done inside the __name__ == "__main__": block
    # and before any other multiprocessing or CUDA calls.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # The start method can only be set once

    ## Start the training, this should take around 15 minutes
    register_vizdoom_components()

    env = "doom_health_gathering_supreme"
    train_the_agent(env)
    enjoy_the_agent(env)
    visualize_play()
