from gymnasium.envs.registration import register

from flight_sim.env import FlightDodgeEnv

register(
    id="FlightDodge-v0",
    entry_point="flight_sim.env:FlightDodgeEnv",
    max_episode_steps=2000,
)

__all__ = ["FlightDodgeEnv"]
