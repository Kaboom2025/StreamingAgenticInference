"""Tests for GridWorld environment."""

import pytest
from unittest.mock import Mock, MagicMock
from streamagent.engine.interfaces import Action, Observation, ObsInjectorProtocol
from streamagent.env.gridworld import GridWorld
from streamagent.env.scenarios import GridCell, Scenario, load_scenarios


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_scenario():
    """4x4 grid with no walls, start at (0,0), goal at (3,3)."""
    return Scenario(
        name="simple",
        width=4,
        height=4,
        start=GridCell(0, 0),
        goal=GridCell(3, 3),
        walls=frozenset(),
        max_steps=100,
    )


@pytest.fixture
def scenario_with_walls():
    """4x4 grid with walls, start at (0,0), goal at (3,3)."""
    walls = frozenset([GridCell(1, 0), GridCell(1, 1), GridCell(1, 2)])
    return Scenario(
        name="with_walls",
        width=4,
        height=4,
        start=GridCell(0, 0),
        goal=GridCell(3, 3),
        walls=walls,
        max_steps=100,
    )


@pytest.fixture
def gridworld(simple_scenario):
    """GridWorld instance with simple scenario."""
    return GridWorld(simple_scenario)


@pytest.fixture
def mock_injector():
    """Mock ObsInjectorProtocol for testing."""
    return Mock(spec=ObsInjectorProtocol)


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------


def test_reset_places_agent_at_start(gridworld, simple_scenario):
    """Agent should be placed at scenario start position after reset."""
    gridworld.reset()
    assert gridworld.current_position == simple_scenario.start


def test_reset_returns_observation_type_gridworld(gridworld):
    """Reset should return observation with type='gridworld'."""
    obs = gridworld.reset()
    assert isinstance(obs, Observation)
    assert obs.type == "gridworld"
    assert len(obs.content) > 0


# ---------------------------------------------------------------------------
# Step - Movement tests
# ---------------------------------------------------------------------------


def test_step_move_north_updates_position(gridworld, simple_scenario):
    """Moving north should decrement y coordinate."""
    # Start at (0, 1) to be able to move north
    gw = GridWorld(
        Scenario(
            name="test",
            width=4,
            height=4,
            start=GridCell(0, 1),
            goal=GridCell(3, 3),
            walls=frozenset(),
            max_steps=100,
        )
    )
    gw.reset()
    initial_y = gw.current_position.y
    action = Action(command="move north", raw='<act cmd="move north"/>')
    obs, reward, done = gw.step(action)
    assert gw.current_position.y < initial_y
    assert not done


def test_step_move_south_updates_position(gridworld, simple_scenario):
    """Moving south should increment y coordinate."""
    gridworld.reset()
    initial_y = gridworld.current_position.y
    action = Action(command="move south", raw='<act cmd="move south"/>')
    obs, reward, done = gridworld.step(action)
    assert gridworld.current_position.y > initial_y
    assert not done


def test_step_move_east_updates_position(gridworld, simple_scenario):
    """Moving east should increment x coordinate."""
    gridworld.reset()
    initial_x = gridworld.current_position.x
    action = Action(command="move east", raw='<act cmd="move east"/>')
    obs, reward, done = gridworld.step(action)
    assert gridworld.current_position.x > initial_x
    assert not done


def test_step_move_west_updates_position(gridworld, simple_scenario):
    """Moving west should decrement x coordinate."""
    gridworld.reset()
    # Start at (0, 0), move east to (1, 0), then west to (0, 0)
    gridworld.step(Action(command="move east", raw='<act cmd="move east"/>'))
    initial_x = gridworld.current_position.x
    action = Action(command="move west", raw='<act cmd="move west"/>')
    obs, reward, done = gridworld.step(action)
    assert gridworld.current_position.x < initial_x


# ---------------------------------------------------------------------------
# Step - Collision tests
# ---------------------------------------------------------------------------


def test_step_blocked_by_wall_stays_in_place(scenario_with_walls):
    """Agent should not move through walls."""
    gw = GridWorld(scenario_with_walls)
    gw.reset()
    initial_pos = gw.current_position
    # Try to move east into wall at (1, 0)
    action = Action(command="move east", raw='<act cmd="move east"/>')
    obs, reward, done = gw.step(action)
    assert gw.current_position == initial_pos


def test_step_blocked_by_boundary_stays_in_place(gridworld):
    """Agent should not move outside grid boundaries."""
    gridworld.reset()
    # Agent starts at (0, 0), can't move west or north
    initial_pos = gridworld.current_position
    action = Action(command="move west", raw='<act cmd="move west"/>')
    obs, reward, done = gridworld.step(action)
    assert gridworld.current_position == initial_pos


def test_step_blocked_by_y_boundary(gridworld):
    """Agent should not move beyond y boundary at max."""
    gw = GridWorld(
        Scenario(
            name="boundary_test",
            width=4,
            height=4,
            start=GridCell(0, 3),
            goal=GridCell(1, 3),
            walls=frozenset(),
            max_steps=100,
        )
    )
    gw.reset()
    initial_pos = gw.current_position
    # Try to move south (south from y=3 is out of bounds)
    action = Action(command="move south", raw='<act cmd="move south"/>')
    obs, reward, done = gw.step(action)
    assert gw.current_position == initial_pos


# ---------------------------------------------------------------------------
# Step - Reward tests
# ---------------------------------------------------------------------------


def test_step_normal_move_reward_is_0(gridworld):
    """Successful move to empty cell should have reward 0."""
    gridworld.reset()
    action = Action(command="move east", raw='<act cmd="move east"/>')
    obs, reward, done = gridworld.step(action)
    assert reward == 0.0
    assert not done


def test_step_blocked_move_reward_is_minus_0_1(gridworld):
    """Blocked move should have reward -0.1."""
    gridworld.reset()
    initial_pos = gridworld.current_position
    action = Action(command="move west", raw='<act cmd="move west"/>')
    obs, reward, done = gridworld.step(action)
    assert reward == -0.1
    assert gridworld.current_position == initial_pos


# ---------------------------------------------------------------------------
# Step - Goal tests
# ---------------------------------------------------------------------------


def test_step_reach_goal_returns_done_true(gridworld, simple_scenario):
    """Reaching goal should return done=True."""
    gw = GridWorld(
        Scenario(
            name="short",
            width=4,
            height=4,
            start=GridCell(0, 0),
            goal=GridCell(1, 0),
            walls=frozenset(),
            max_steps=100,
        )
    )
    gw.reset()
    action = Action(command="move east", raw='<act cmd="move east"/>')
    obs, reward, done = gw.step(action)
    assert done is True


def test_step_reach_goal_reward_is_1(gridworld, simple_scenario):
    """Reaching goal should have reward 1.0."""
    gw = GridWorld(
        Scenario(
            name="short",
            width=4,
            height=4,
            start=GridCell(0, 0),
            goal=GridCell(1, 0),
            walls=frozenset(),
            max_steps=100,
        )
    )
    gw.reset()
    action = Action(command="move east", raw='<act cmd="move east"/>')
    obs, reward, done = gw.step(action)
    assert reward == 1.0


# ---------------------------------------------------------------------------
# Step - Look command tests
# ---------------------------------------------------------------------------


def test_step_look_no_position_change(gridworld):
    """Look command should not change position."""
    gridworld.reset()
    initial_pos = gridworld.current_position
    action = Action(command="look", raw='<act cmd="look"/>')
    obs, reward, done = gridworld.step(action)
    assert gridworld.current_position == initial_pos
    assert not done


def test_step_look_returns_observation(gridworld):
    """Look command should return valid observation."""
    gridworld.reset()
    action = Action(command="look", raw='<act cmd="look"/>')
    obs, reward, done = gridworld.step(action)
    assert isinstance(obs, Observation)
    assert obs.type == "gridworld"
    assert len(obs.content) > 0


# ---------------------------------------------------------------------------
# Step - Max steps tests
# ---------------------------------------------------------------------------


def test_step_exceeds_max_steps_returns_done(gridworld):
    """Exceeding max_steps should return done=True."""
    gw = GridWorld(
        Scenario(
            name="tiny",
            width=4,
            height=4,
            start=GridCell(0, 0),
            goal=GridCell(10, 10),  # unreachable goal
            walls=frozenset(),
            max_steps=3,
        )
    )
    gw.reset()
    # Use 3 steps worth of moves
    for i in range(3):
        action = Action(command="move east", raw='<act cmd="move east"/>')
        obs, reward, done = gw.step(action)
    assert done is True


# ---------------------------------------------------------------------------
# Injector tests
# ---------------------------------------------------------------------------


def test_register_injector_puts_obs_on_step(gridworld, mock_injector):
    """Registering injector should call put() on each step observation."""
    gridworld.register_injector(mock_injector)
    gridworld.reset()
    # Reset should call put
    assert mock_injector.put.called
    # Reset call count
    mock_injector.reset_mock()
    action = Action(command="move east", raw='<act cmd="move east"/>')
    obs, reward, done = gridworld.step(action)
    # Step should also call put
    assert mock_injector.put.called
    call_args = mock_injector.put.call_args[0][0]
    assert isinstance(call_args, Observation)
    assert call_args.type == "gridworld"


# ---------------------------------------------------------------------------
# Render tests
# ---------------------------------------------------------------------------


def test_render_contains_agent_marker(gridworld):
    """Render should contain 'A' for agent position."""
    gridworld.reset()
    rendered = gridworld.render()
    assert "A" in rendered


def test_render_contains_goal_marker(gridworld):
    """Render should contain 'G' for goal position."""
    gridworld.reset()
    rendered = gridworld.render()
    assert "G" in rendered


def test_render_contains_walls(scenario_with_walls):
    """Render should contain '#' for walls."""
    gw = GridWorld(scenario_with_walls)
    gw.reset()
    rendered = gw.render()
    assert "#" in rendered


# ---------------------------------------------------------------------------
# Scenarios tests
# ---------------------------------------------------------------------------


def test_load_scenarios_returns_at_least_5():
    """load_scenarios() should return at least 5 default scenarios."""
    scenarios = load_scenarios()
    assert len(scenarios) >= 5
    # All should be valid Scenario objects
    for scenario in scenarios:
        assert isinstance(scenario, Scenario)
        assert scenario.width > 0
        assert scenario.height > 0
        assert scenario.start is not None
        assert scenario.goal is not None


def test_load_scenarios_default_none():
    """load_scenarios(None) should return default scenarios."""
    scenarios = load_scenarios(None)
    assert len(scenarios) >= 5


def test_load_scenarios_with_path():
    """load_scenarios(path) should return default scenarios (path ignored for now)."""
    scenarios = load_scenarios("/some/path.json")
    assert len(scenarios) >= 5


def test_scenarios_are_solvable(simple_scenario):
    """Default scenarios should be solvable (start != goal)."""
    assert simple_scenario.start != simple_scenario.goal


def test_gridcell_immutable():
    """GridCell should be immutable."""
    cell = GridCell(1, 2)
    with pytest.raises(Exception):  # frozen dataclass
        cell.x = 5  # type: ignore[misc]


def test_scenario_immutable():
    """Scenario should be immutable."""
    scenario = Scenario(
        name="test",
        width=4,
        height=4,
        start=GridCell(0, 0),
        goal=GridCell(3, 3),
        walls=frozenset(),
        max_steps=100,
    )
    with pytest.raises(Exception):
        scenario.name = "other"  # type: ignore[misc]
