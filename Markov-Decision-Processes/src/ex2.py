import ext_plant
import numpy as np

id = ["000000000"]

class Controller:
    """This class is a controller for the ext_plant game."""

    def __init__(self, game: ext_plant.Game):
        """Initialize controller for given game model."""
        self.original_game = game
  
    def choose_next_action(self, state):
        """ Choose the next action given a state."""
        possible_moves = []

        (robots, plants, taps, total_water_need) = state
        capacities = self.original_game.get_capacities()

        directions = [((0, 1), "RIGHT"), ((0, -1), "LEFT"), ((-1, 0), "UP"), ((1, 0), "DOWN")]
        for robot_id, (r, c), load in robots:
            remaining_capacity = capacities[robot_id] - load
            for (dr, dc), action_name in directions:
                destination = (r + dr, c + dc)
                if (all(cords != destination for (_, cords, __) in robots) and
                    destination not in self.original_game.walls and
                    0 <= r + dr < self.original_game.rows and
                    0 <= c + dc < self.original_game.cols):
                    possible_moves.append(f"{action_name}({robot_id})")
            if load > 0:
                plant_in_current_location = next(((pos, need) for (pos, need) in plants if (r, c) == pos and need > 0), None)
                if plant_in_current_location is not None:
                    possible_moves.append(f"POUR({robot_id})")
            if remaining_capacity > 0:
                tap_in_current_location = next(((pos, available_water) for (pos, available_water) in taps if (r, c) == pos and available_water > 0), None)
                if tap_in_current_location is not None:
                    possible_moves.append(f"LOAD({robot_id})")
        return np.random.choice(np.array(possible_moves))
