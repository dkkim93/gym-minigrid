from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """
    def __init__(
        self,
        size=8,
        agent_start_pos=(3, 3),
        agent_start_dir=0,
    ):
        self.size = size + 2  # Considering the surrounding walls
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = None
        self.mission = ""  # Dummy variable

        super().__init__(
            grid_size=self.size,
            max_steps=4 * self.size * self.size,
            see_through_walls=True  # Set this to True for maximum speed
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Set goal
        assert self.goal_pos is not None, "Goal position is None. Call env.reset_task() before env.reset()"
        self.put_obj(Goal(), self.goal_pos[0] + 1, self.goal_pos[1] + 1)

    def _reward(self):
        dist = np.linalg.norm(np.array(self.agent_pos) - self.goal_pos)
        return -dist

    def reset_task(self, task):
        if task is None:
            task = self.np_random.randint(0, self.size - 2, size=(2,))
        assert len(task) == 2, "Must in format of (col, row)"
        assert task[0] >= 0 and task[0] < self.size - 2
        assert task[1] >= 0 and task[1] < self.size - 2
        self.goal_pos = np.array(task)

    def sample_tasks(self, num_tasks):
        tasks = self.np_random.randint(0, self.size - 2, size=(num_tasks, 2))
        # tasks = self.np_random.randint(1, 2, size=(num_tasks, 2))
        return tasks
    

class EmptyEnv5x5(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=5, **kwargs)


register(
    id='MiniGrid-Empty-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyEnv5x5'
)
