import numpy as np
from gym_minigrid.minigrid import Key
from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.register import register


class Unlock(RoomGrid):
    def __init__(self, seed=None):
        room_size = 7
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

        # Make sure the two rooms are directly connected by a locked door
        self.door_pos = (6, 1)
        door, _ = self.add_door(0, 0, pos=self.door_pos, door_idx=0, locked=True, color="red")
        self.door = door

        # Set key
        assert self.key_pos is not None, "Key position is None. Call env.reset_task() before env.reset()"
        self.key = Key(self.door.color)
        self.put_obj(self.key, self.key_pos[0], self.key_pos[1])

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.carrying is None:
            dist = np.linalg.norm(np.array(self.agent_pos) - np.array(self.key.cur_pos))
            reward = -dist
        else:
            dist = np.linalg.norm(np.array(self.agent_pos) - np.array(self.door_pos))
            reward = -dist

            if np.array_equal(self.agent_pos, self.door_pos) and self.door.is_open is True:
                reward = 1.

        done = False
        if action == self.actions.toggle:
            if self.door.is_open:
                done = True

        if self.step_count >= self.max_steps:
            done = True

        return obs, reward, done, info

    def reset_task(self, task):
        assert len(task) == 2, "Must in format of (col, row)"
        assert task[0] >= 1 and task[0] < 6
        assert task[1] >= 1 and task[1] < 6
        self.key_pos = np.array(task)

    def sample_tasks(self, num_tasks):
        tasks = self.np_random.randint(2, 6, size=(num_tasks, 2))
        return tasks


register(
    id='MiniGrid-Unlock-v0',
    entry_point='gym_minigrid.envs:Unlock'
)
