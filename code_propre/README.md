# Explanation of the code

## Linear feedback

In the file `run_linear_basic.py`, we just apply the action with a gain and the observations.

## Utils

In the `utils.py` file, the function compute_msfos is used to compute by dichotomy the msfos with respect to the algorithm we wrote in the report :

```plaintext
1. Initialize lower_bound, upper_bound, tolerance, num_trials
2. While (upper_bound - lower_bound > tolerance):
     a. mid_force = (lower_bound + upper_bound) / 2
     b. success_probability = EstimateSuccessProbability(mid_force, num_trials)
     c. If (success_probability > 2/3):
            lower_bound = mid_force
        Else:
            upper_bound = mid_force
3. Calculate MSFOS = (lower_bound + upper_bound) / 2
4. Return MSFOS
```

## New Wrapper

The file `train.py` that is in the ppo_balancer by default has no application of forces to the torso. Therefore, we need to add a new wrapper to the environment, it is in the `train.py` in this repertory.

```python
class VelocityEnvWrapper(gymnasium.Wrapper):
    """
    A custom wrapper for the velocity environment.

    This wrapper allows us to modify or log observations, rewards,
    or interactions with the environment.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.count = 0
        self.counttotal = 0
        self.truncated = 0
        self.max_steps = 3000
        self.force_schedule = []
        


    def reset(self, **kwargs):
        """
        Reset the environment and modify the initial observation if needed.
        """
        if not self.truncated:
            print("truncated ! ")
            print(self.force_schedule)
        else: print("resisted",self.force_schedule)
        obs, info = self.env.reset(**kwargs)
        self.count = 0
        self.scales = [0, 5, 0, 9, 0, 12, 0]  # Corresponding scales for the random values
        self.num_intervals = 3
        
        
        # boundaries = sorted(np.random.choice(range(1, self.max_steps), self.num_intervals - 1, replace=False))
        boundaries = [0,200,400,800, 1000, 1700, 1900, self.max_steps]


        # Generate the force schedule with random steps
        self.force_schedule = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1] - 1
            scale = self.scales[min(i, len(self.scales) - 1)]
            force = [np.clip(np.random.normal(loc=scale/3, scale=scale/2),-scale,scale), 0.0, 0.0]
            self.force_schedule.append((start, end, force))
        return obs, info
    
    def apply_external_force(self,force):
        bullet_action = {
            "external_forces": {
                "torso": {
                    "force": force,
                    "local": False,
                }
            }
        }
        self.env.bullet_extra(bullet_action)

    def step(self, action):
        """
        Execute a step in the environment and modify the results if needed.
        """
        
        force = self._get_force_for_step()

        # Apply the determined force to the robot
        self.apply_external_force(force)

        obs, reward, done, truncated, info = self.env.step(action)

        # reward = self.modify_reward(reward)

        self.count += 1
        self.counttotal+=1
        self.truncated = done
        return obs, reward, done, truncated, info
    
    def _get_force_for_step(self):
        """
        Get the force to apply based on the current step from the force schedule.
        """
        for start_step, end_step, force in self.force_schedule:
            if start_step <= self.count <= end_step:
                return np.array(force)
        return np.array([0.0, 0.0, 0.0])

    def modify_reward(self, reward):
        """
        Modify the reward before returning it.
        Example: Apply a scaling factor.
        """
        return reward
```
This is the wrapper for the best policy we had with a 10.3 N MSFOS. We compute a force schedule, given the time boundaries and the scales associated (see the reset method).

After the training, we get a policy in the form of .zip file then we launch `compute_msfos_policy.py`.