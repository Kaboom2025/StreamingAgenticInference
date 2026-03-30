"""ASCII renderer for GridWorld environment."""

from streamagent.env.scenarios import GridCell, Scenario


def render_grid(scenario: Scenario, agent_pos: GridCell) -> str:
    """Render GridWorld as ASCII art.

    Args:
        scenario: The scenario definition containing grid dimensions and walls.
        agent_pos: Current agent position.

    Returns:
        ASCII string representation with:
        - '#' for grid borders and walls
        - 'A' for agent position
        - 'G' for goal
        - '.' for empty cells
    """
    lines = []

    # Top border
    lines.append("#" * (scenario.width + 2))

    # Grid rows
    for y in range(scenario.height):
        row = "#"  # left border
        for x in range(scenario.width):
            cell = GridCell(x, y)
            if cell == agent_pos:
                row += "A"
            elif cell == scenario.goal:
                row += "G"
            elif cell in scenario.walls:
                row += "#"
            else:
                row += "."
        row += "#"  # right border
        lines.append(row)

    # Bottom border
    lines.append("#" * (scenario.width + 2))

    return "\n".join(lines)
