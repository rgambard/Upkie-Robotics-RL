import gymnasium as gym
import numpy as np
import upkie.envs

upkie.envs.register()

with gym.make("UpkieGroundVelocity-v3", frequency=200.0) as env:
    torso_force_in_world = np.zeros(3)
    bullet_action = {
        "external_forces": {
            "torso": {
                "force": torso_force_in_world,
                "local": False,
            }
        }
    }
    observation, _ = env.reset()
    gain = np.array([20.0, 0.0, 0.0, 0.0])
    for step in range(1_000_000):
        action = gain.dot(observation).reshape((1,))
        observation, reward, terminated, truncated, _ = env.step(action)
        if step>=100 and step<=300:
            torso_force_in_world[0] = 10
        else:
            torso_force_in_world[0] = 0
        env.bullet_extra(bullet_action)
        if terminated or truncated:
            observation,_ = env.reset()
