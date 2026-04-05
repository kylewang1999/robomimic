try:
    from .obs_core import EncoderCore, Randomizer
except ModuleNotFoundError:
    # Allow direct imports of lightweight submodules such as
    # `diffusion_policy_nets` in environments missing optional vision deps.
    EncoderCore = None
    Randomizer = None
