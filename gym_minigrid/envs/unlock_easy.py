import random
import numpy as np
from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.register import register


class UnlockEasy(RoomGrid):
    def __init__(self, seed=None):
        room_size = 7

        self.possible_tasks = [
            # Bottom
            np.array([1, 6]),
            np.array([2, 6]),
            np.array([3, 6]),
            np.array([4, 6]),
            np.array([5, 6]),
            # Right
            np.array([6, 1]),
            np.array([6, 2]),
            np.array([6, 3]),
            np.array([6, 4]),
            np.array([6, 5]),
            # Left
            np.array([0, 1]),
            np.array([0, 2]),
            np.array([0, 3]),
            np.array([0, 4]),
            np.array([0, 5]),
            # Top
            np.array([1, 0]),
            np.array([2, 0]),
            np.array([3, 0]),
            np.array([4, 0]),
            np.array([5, 0])
        ]

        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            max_steps=8 * room_size**2,
            seed=seed
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        super()._gen_grid(width, height)

        # Set door
        assert self.door_pos is not None, "Door position is None. Call env.reset_task() before env.reset()"
        door, _ = self.add_door(0, 0, pos=self.door_pos, door_idx=0, locked=True, color="red")
        self.door = door
        assert door.is_locked is True and door.is_open is False, "Initially, door should be locked"

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # Compute reward
        if self.door.is_open:
            reward = 1.
        else:
            dist = np.linalg.norm(np.array(self.agent_pos) - np.array(self.door_pos))
            reward = -dist

        # Compute done
        done = False
        if self.door.is_open:
            done = True

        if self.step_count >= self.max_steps:
            done = True

        return obs, reward, done, info

    def reset_task(self, task):
        if task is None:
            task = self.possible_tasks[0]
        self.door_pos = np.array(task)

    def sample_tasks(self, num_tasks):
        tasks = random.choices(self.possible_tasks, k=num_tasks)
        return tasks


register(
    id='MiniGrid-Unlock-Easy-v0',
    entry_point='gym_minigrid.envs:UnlockEasy'
)
