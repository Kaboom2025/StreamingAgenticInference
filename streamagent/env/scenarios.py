"""Scenario definitions and loaders for GridWorld environment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GridCell:
    """Immutable grid cell coordinate."""

    x: int
    y: int


@dataclass(frozen=True)
class Scenario:
    """Immutable scenario definition for GridWorld."""

    name: str
    width: int
    height: int
    start: GridCell
    goal: GridCell
    walls: frozenset[GridCell]
    max_steps: int = 100


def load_scenarios(path: str | None = None) -> list[Scenario]:
    """Load GridWorld scenarios.

    If path is None, return a default set of hardcoded scenarios.

    Args:
        path: Optional path to scenario file. Currently unused; always returns defaults.

    Returns:
        List of at least 5 solvable scenarios with varying difficulty.
    """
    if path is not None:
        # Future: load from file if path is provided
        pass

    return [
        # Simple 4x4 grid, no obstacles
        Scenario(
            name="simple_4x4",
            width=4,
            height=4,
            start=GridCell(0, 0),
            goal=GridCell(3, 3),
            walls=frozenset(),
            max_steps=100,
        ),
        # 6x6 grid with scattered walls
        Scenario(
            name="scattered_6x6",
            width=6,
            height=6,
            start=GridCell(0, 0),
            goal=GridCell(5, 5),
            walls=frozenset(
                [
                    GridCell(2, 1),
                    GridCell(3, 2),
                    GridCell(2, 4),
                    GridCell(4, 3),
                ]
            ),
            max_steps=150,
        ),
        # 8x8 grid with wall maze
        Scenario(
            name="maze_8x8",
            width=8,
            height=8,
            start=GridCell(0, 0),
            goal=GridCell(7, 7),
            walls=frozenset(
                [
                    GridCell(1, 1),
                    GridCell(1, 2),
                    GridCell(1, 3),
                    GridCell(3, 3),
                    GridCell(3, 4),
                    GridCell(3, 5),
                    GridCell(5, 5),
                    GridCell(5, 6),
                    GridCell(5, 4),
                ]
            ),
            max_steps=200,
        ),
        # 6x6 grid with narrow corridor
        Scenario(
            name="corridor_6x6",
            width=6,
            height=6,
            start=GridCell(0, 2),
            goal=GridCell(5, 2),
            walls=frozenset(
                [
                    GridCell(1, 0),
                    GridCell(1, 1),
                    GridCell(1, 3),
                    GridCell(1, 4),
                    GridCell(1, 5),
                    GridCell(3, 0),
                    GridCell(3, 1),
                    GridCell(3, 3),
                    GridCell(3, 4),
                    GridCell(3, 5),
                ]
            ),
            max_steps=150,
        ),
        # 4x4 grid with complex walls
        Scenario(
            name="complex_4x4",
            width=4,
            height=4,
            start=GridCell(0, 0),
            goal=GridCell(3, 3),
            walls=frozenset([GridCell(1, 0), GridCell(1, 1), GridCell(2, 2)]),
            max_steps=100,
        ),
    ]
