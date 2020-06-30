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
        super()._gen_grid(width, height)

        # Make sure the two rooms are directly connected by a locked door
        self.door_pos = (6, 1)
        door, _ = self.add_door(0, 0, pos=self.door_pos, door_idx=0, locked=True, color="red")
        self.door = door

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.carrying is None:
            dist = np.linalg.norm(np.array(self.agent_pos) - np.array(self.key.cur_pos))
            reward = -dist
        else:
            dist = np.linalg.norm(np.array(self.agent_pos) - np.array(self.door_pos))
            reward = -dist

        done = False
        if action == self.actions.toggle:
            if self.door.is_open:
                done = True

        return obs, reward, done, info

    def _set_task(self, task):
        # Add a key to unlock the door
        self.key_pos = task
        self.key = Key(self.door.color)
        self.put_obj(self.key, self.key_pos[0], self.key_pos[1])


register(
    id='MiniGrid-Unlock-v0',
    entry_point='gym_minigrid.envs:Unlock'
)
