"""GridWorld environment for StreamAgent."""

from streamagent.engine.interfaces import (
    Action,
    Environment,
    Observation,
    ObsInjectorProtocol,
)
from streamagent.env.scenarios import GridCell, Scenario
from streamagent.env.renderer import render_grid


class GridWorld(Environment):
    """2D grid navigation environment for evaluating StreamAgent recovery latency.

    Attributes:
        scenario: The scenario definition (grid size, walls, start, goal).
        current_position: Agent's current position in the grid.
    """

    def __init__(self, scenario: Scenario) -> None:
        """Initialize GridWorld with a scenario.

        Args:
            scenario: Scenario object defining grid, walls, start, and goal positions.
        """
        self.scenario = scenario
        self._position = scenario.start
        self._step_count = 0
        self._injector: ObsInjectorProtocol | None = None

    @property
    def current_position(self) -> GridCell:
        """Get the agent's current position."""
        return self._position

    def reset(self) -> Observation:
        """Reset environment to initial state.

        Returns:
            Observation describing the starting position and goal.
        """
        self._position = self.scenario.start
        self._step_count = 0

        obs = Observation(
            type="gridworld",
            content=self._create_description(),
        )

        if self._injector is not None:
            self._injector.put(obs)

        return obs

    def step(self, action: Action) -> tuple[Observation, float, bool]:
        """Execute action in the environment.

        Commands:
        - "move north": move to (x, y-1) if not blocked
        - "move south": move to (x, y+1) if not blocked
        - "move east": move to (x+1, y) if not blocked
        - "move west": move to (x-1, y) if not blocked
        - "look": no movement, just observe current state

        Rewards:
        - Successful move: 0.0
        - Blocked move: -0.1
        - Goal reached: 1.0
        - Exceeded max_steps: done=True with reward 0.0

        Args:
            action: Action object with command string.

        Returns:
            Tuple of (observation, reward, done).
        """
        self._step_count += 1
        reward = 0.0
        done = False

        # Handle movement commands
        if action.command.startswith("move "):
            direction = action.command.replace("move ", "").strip()
            new_pos = self._get_new_position(direction)

            # Check if move is valid (not blocked by wall or boundary)
            if self._is_valid_position(new_pos):
                self._position = new_pos
                reward = 0.0
            else:
                reward = -0.1

        # Check if goal is reached
        if self._position == self.scenario.goal:
            reward = 1.0
            done = True

        # Check if max steps exceeded
        if self._step_count >= self.scenario.max_steps and not done:
            done = True

        obs = Observation(
            type="gridworld",
            content=self._create_description(),
        )

        if self._injector is not None:
            self._injector.put(obs)

        return obs, reward, done

    def register_injector(self, injector: ObsInjectorProtocol) -> None:
        """Register an observation injector.

        Args:
            injector: ObsInjectorProtocol instance to receive observations.
        """
        self._injector = injector

    def render(self) -> str:
        """Return ASCII art representation of current state.

        Returns:
            String with grid visualization.
        """
        return render_grid(self.scenario, self._position)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _get_new_position(self, direction: str) -> GridCell:
        """Calculate new position for a given direction.

        Args:
            direction: One of "north", "south", "east", "west".

        Returns:
            GridCell with updated coordinates.
        """
        x, y = self._position.x, self._position.y

        if direction == "north":
            y -= 1
        elif direction == "south":
            y += 1
        elif direction == "east":
            x += 1
        elif direction == "west":
            x -= 1

        return GridCell(x, y)

    def _is_valid_position(self, pos: GridCell) -> bool:
        """Check if a position is within bounds and not a wall.

        Args:
            pos: GridCell to validate.

        Returns:
            True if position is walkable, False otherwise.
        """
        # Check bounds
        if pos.x < 0 or pos.x >= self.scenario.width:
            return False
        if pos.y < 0 or pos.y >= self.scenario.height:
            return False

        # Check walls
        if pos in self.scenario.walls:
            return False

        return True

    def _create_description(self) -> str:
        """Create a natural language description of the current state.

        Returns:
            String describing agent position, goal, and grid state.
        """
        pos = self._position
        goal = self.scenario.goal
        distance = abs(pos.x - goal.x) + abs(pos.y - goal.y)

        return (
            f"Agent at ({pos.x}, {pos.y}). Goal at ({goal.x}, {goal.y}). "
            f"Manhattan distance: {distance}. Steps taken: {self._step_count}."
        )
