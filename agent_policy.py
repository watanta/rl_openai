from luxai2021.game.match_controller import ActionSequence
import sys
import time
from functools import partial  # pip install functools

import numpy as np
from gym import spaces
import copy
import random

from luxai2021.env.agent import Agent
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.position import Position


# https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)
def furthest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmax(dist_2)

def smart_transfer_to_nearby(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
    """
    Smart-transfers from the specified unit to a nearby neighbor. Prioritizes any
    nearby carts first, then any worker. Transfers the resource type which the unit
    has most of. Picks which cart/worker based on choosing a target that is most-full
    but able to take the most amount of resources.

    Args:
        team ([type]): [description]
        unit_id ([type]): [description]

    Returns:
        Action: Returns a TransferAction object, even if the request is an invalid
                transfer. Use TransferAction.is_valid() to check validity.
    """

    # Calculate how much resources could at-most be transferred
    resource_type = None
    resource_amount = 0
    target_unit = None

    if unit != None:
        for type, amount in unit.cargo.items():
            if amount > resource_amount:
                resource_type = type
                resource_amount = amount

        # Find the best nearby unit to transfer to
        unit_cell = game.map.get_cell_by_pos(unit.pos)
        adjacent_cells = game.map.get_adjacent_cells(unit_cell)

        
        for c in adjacent_cells:
            for id, u in c.units.items():
                # Apply the unit type target restriction
                if target_type_restriction == None or u.type == target_type_restriction:
                    if u.team == team:
                        # This unit belongs to our team, set it as the winning transfer target
                        # if it's the best match.
                        if target_unit is None:
                            target_unit = u
                        else:
                            # Compare this unit to the existing target
                            if target_unit.type == u.type:
                                # Transfer to the target with the least capacity, but can accept
                                # all of our resources
                                if( u.get_cargo_space_left() >= resource_amount and 
                                    target_unit.get_cargo_space_left() >= resource_amount ):
                                    # Both units can accept all our resources. Prioritize one that is most-full.
                                    if u.get_cargo_space_left() < target_unit.get_cargo_space_left():
                                        # This new target it better, it has less space left and can take all our
                                        # resources
                                        target_unit = u
                                    
                                elif( target_unit.get_cargo_space_left() >= resource_amount ):
                                    # Don't change targets. Current one is best since it can take all
                                    # the resources, but new target can't.
                                    pass
                                    
                                elif( u.get_cargo_space_left() > target_unit.get_cargo_space_left() ):
                                    # Change targets, because neither target can accept all our resources and 
                                    # this target can take more resources.
                                    target_unit = u
                            elif u.type == Constants.UNIT_TYPES.CART:
                                # Transfer to this cart instead of the current worker target
                                target_unit = u
    
    # Build the transfer action request
    target_unit_id = None
    if target_unit is not None:
        target_unit_id = target_unit.id

        # Update the transfer amount based on the room of the target
        if target_unit.get_cargo_space_left() < resource_amount:
            resource_amount = target_unit.get_cargo_space_left()
    
    return TransferAction(team, unit_id, target_unit_id, resource_type, resource_amount)

########################################################################################################################
# This is the Agent that you need to design for the competition
########################################################################################################################
class AgentPolicy(Agent):
    def __init__(self, mode="train", model=None) -> None:
        """
        Arguments:
            mode: "train" or "inference", which controls if this agent is for training or not.
            model: The pretrained model, or if None it will operate in training mode.
        """
        super().__init__()
        self.model = model
        self.mode = mode
        
        self.stats = None
        self.stats_last_game = None

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.actionSpaceMapUnits = [
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            SpawnCityAction,
            #PillageAction,
        ]


        self.action_space = spaces.Discrete(len(self.actionSpaceMapUnits))
        

        # Observation space: (Basic minimum for a miner agent)
        # Object:
        #   1x is worker
        #   1x is cart
        #   1x is citytile
        #
        #   5x direction_nearest_wood
        #   1x distance_nearest_wood
        #   1x amount
        #
        #   5x direction_nearest_coal
        #   1x distance_nearest_coal
        #   1x amount
        #
        #   5x direction_nearest_uranium
        #   1x distance_nearest_uranium
        #   1x amount
        #
        #   5x direction_nearest_city
        #   1x distance_nearest_city
        #   1x amount of fuel
        #
        #   28x (the same as above, but direction, distance, and amount to the furthest of each)
        #
        #   5x direction_nearest_worker
        #   1x distance_nearest_worker
        #   1x amount of cargo
        # Unit:
        #   1x cargo size
        # State:
        #   1x is night
        #   1x percent of game done
        #   2x citytile counts [cur player, opponent]
        #   2x worker counts [cur player, opponent]
        #   2x cart counts [cur player, opponent]
        #   1x research points [cur player]
        #   1x researched coal [cur player]
        #   1x researched uranium [cur player]
        self.observation_shape = (20, 32, 32)
        self.observation_space = spaces.Box(low=0, high=1, shape=
        self.observation_shape, dtype=np.float16)

        self.object_nodes = {}

    def get_agent_type(self):
        """
        Returns the type of agent. Use AGENT for inference, and LEARNING for training a model.
        """
        if self.mode == "train":
            return Constants.AGENT_TYPE.LEARNING
        else:
            return Constants.AGENT_TYPE.AGENT

    def get_observation(self, game, unit, city_tile, input_team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city
        """
        width = game.map.width
        height = game.map.height
        x_shift = (32 - width) // 2
        y_shift = (32 - height) // 2
        cities = {}

        b = np.zeros((20, 32, 32), dtype=np.float16)

        # input ownself
        # if input is unit
        if unit is not None:
            x = unit.pos.x + x_shift
            y = unit.pos.y + y_shift
            b[:2, x, y] = (
                1,
                (unit.cargo["wood"] + unit.cargo["coal"] + unit.cargo["uranium"]) / 100
            )

        # Units
        for team in game.state["teamStates"]:
            for team_unit in game.state["teamStates"][team]["units"].values():
                x = team_unit.pos.x + x_shift
                y = team_unit.pos.y + y_shift
                idx = 2 + abs(team - input_team) * 3
                b[idx:idx + 3, x, y] = (
                    1,
                    team_unit.cooldown / 6,
                    (team_unit.cargo["wood"] + team_unit.cargo["coal"] + team_unit.cargo["uranium"]) / 100
                )
        
        # Citytiles
        for city in game.cities.values():
            for city_cell in city.city_cells:
                x = city_cell.pos.x + x_shift
                y = city_cell.pos.y + y_shift
                idx = 8 + (city.team - input_team) * 2
                b[idx:idx + 2, x, y] = (
                    1,
                    min(city.fuel / city.get_light_upkeep(), 10) / 10
                )

        # Resources
        if "wood" in game.map.resources_by_type:
            for wood_tile in game.map.resources_by_type["wood"]:
                x = wood_tile.pos.x + x_shift
                y = wood_tile.pos.y + y_shift
                amt = wood_tile.resource.amount
                b[12, x, y] = amt / 800
        if "coal" in game.map.resources_by_type:
            for coal_tile in game.map.resources_by_type["coal"]:
                x = coal_tile.pos.x + x_shift
                y = coal_tile.pos.y + y_shift
                amt = coal_tile.resource.amount
                b[13, x, y] = amt / 800
        if "uranium" in game.map.resources_by_type:
            for uranium_tile in game.map.resources_by_type["uranium"]:
                x = uranium_tile.pos.x + x_shift
                y = uranium_tile.pos.y + y_shift
                amt = uranium_tile.resource.amount
                b[14, x, y] = amt / 800
        
        # Resource point
        for team in game.state["teamStates"]:
            rp = game.state["teamStates"][team]["researchPoints"]
            b[15 + abs(team - input_team), :] = min(rp, 200) / 200

        # Day/Night Cycle
        b[17, :] = game.state["turn"] % 40 / 40
        # Turns
        b[18, :] = game.state["turn"] / 360
        # Map Size
        b[19, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

        return b


    def action_code_to_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        Returns: An action.
        """
        # Map actionCode index into to a constructed Action object
        try:
            x = None
            y = None
            if city_tile is not None:
                x = city_tile.pos.x
                y = city_tile.pos.y
            elif unit is not None:
                x = unit.pos.x
                y = unit.pos.y
            
            if city_tile != None:
                action =  self.actionSpaceMapCities[action_code%len(self.actionSpaceMapCities)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )

                # If the city action is invalid, default to research action automatically
                if not action.is_valid(game, actions_validated=[]):
                    action = ResearchAction(
                        game=game,
                        unit_id=unit.id if unit else None,
                        unit=unit,
                        city_id=city_tile.city_id if city_tile else None,
                        citytile=city_tile,
                        team=team,
                        x=x,
                        y=y
                    )
            else:
                action =  self.actionSpaceMapUnits[action_code%len(self.actionSpaceMapUnits)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            
            return action
        except Exception as e:
            # Not a valid action
            print(e)
            return None

    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        """
        if city_tile is None:
            action = self.action_code_to_action(action_code, game, unit, city_tile, team)
            self.match_controller.take_action(action)
        else:
            unit_count = len(game.state["teamStates"][0]["units"])
            rp = game.state["teamStates"][team]["researchPoints"]
            researched_uranium = False
            if rp >= 200:
                researched_uranium = True
            city_tile_count = 0
            for tmp_city_tile in game.cities:
                if game.cities[tmp_city_tile].team == team:
                    city_tile_count += 1
            x = city_tile.pos.x
            y = city_tile.pos.y
            if unit_count < city_tile_count: 
                action = SpawnWorkerAction(
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
                pass
            elif not researched_uranium:
                action = ResearchAction(
                                game=game,
                                unit_id=unit.id if unit else None,
                                unit=unit,
                                city_id=city_tile.city_id if city_tile else None,
                                citytile=city_tile,
                                team=team,
                                x=x,
                                y=y
                            )
            self.match_controller.take_action(action)

    
    def game_start(self, game):
        """
        This funciton is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        self.last_generated_fuel = game.stats["teamStats"][self.team]["fuelGenerated"]
        self.last_resources_collected = copy.deepcopy(game.stats["teamStats"][self.team]["resourcesCollected"])
        if self.stats != None:
            self.stats_last_game =  self.stats
        self.stats = {
            "rew/r_total": 0,
            "rew/r_wood": 0,
            "rew/r_coal": 0,
            "rew/r_uranium": 0,
            "rew/r_research": 0,
            "rew/r_city_tiles_end": 0,
            "rew/r_fuel_collected":0,
            "rew/r_units":0,
            "rew/r_city_tiles":0,
#             "rew/r_win":0,
#             "rew/r_city_tiles_more_opponent":0,
            "game/turns": 0,
            "game/research": 0,
            "game/unit_count": 0,
            "game/cart_count": 0,
            "game/city_count": 0,
            "game/city_tiles": 0,
            "game/wood_rate_mined": 0,
            "game/coal_rate_mined": 0,
            "game/uranium_rate_mined": 0,
        }
        self.is_last_turn = False

        # Calculate starting map resources
        type_map = {
            Constants.RESOURCE_TYPES.WOOD: "WOOD",
            Constants.RESOURCE_TYPES.COAL: "COAL",
            Constants.RESOURCE_TYPES.URANIUM: "URANIUM",
        }

        self.fuel_collected_last = 0
        self.fuel_start = {}
        self.fuel_last = {}
        for type, type_upper in type_map.items():
            self.fuel_start[type] = 0
            self.fuel_last[type] = 0
            for c in game.map.resources_by_type[type]:
                self.fuel_start[type] += c.resource.amount * game.configs["parameters"]["RESOURCE_TO_FUEL_RATE"][type_upper]

        self.research_last = 0
        self.units_last = 0
        self.city_tiles_last = 0

    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game.
        """
        if is_game_error:
            # Game environment step failed, assign a game lost reward to not incentivise this
            print("Game failed due to error")
            return -1.0

        if not is_new_turn and not is_game_finished:
            # Only apply rewards at the start of each turn
            return 0

        # Get some basic stats
        unit_count = len(game.state["teamStates"][self.team % 2]["units"])
        cart_count = 0
        for id, u in game.state["teamStates"][self.team % 2]["units"].items():
            if u.type == Constants.UNIT_TYPES.CART:
                cart_count += 1

        unit_count_opponent = len(game.state["teamStates"][(self.team + 1) % 2]["units"])
        research = min(game.state["teamStates"][self.team]["researchPoints"], 200.0) # Cap research points at 200
        city_count = 0
        city_count_opponent = 0
        city_tile_count = 0
        city_tile_count_opponent = 0
        for city in game.cities.values():
            if city.team == self.team:
                city_count += 1
            else:
                city_count_opponent += 1

            for cell in city.city_cells:
                if city.team == self.team:
                    city_tile_count += 1
                else:
                    city_tile_count_opponent += 1
        
        # Basic stats
        self.stats["game/research"] = research
        self.stats["game/city_tiles"] = city_tile_count
        self.stats["game/city_count"] = city_count
        self.stats["game/unit_count"] = unit_count
        self.stats["game/cart_count"] = cart_count
        self.stats["game/turns"] = game.state["turn"]

        rewards = {}

        # Give up to 1.0 reward for each resource based on % of total mined.
        type_map = {
            Constants.RESOURCE_TYPES.WOOD: "WOOD",
            Constants.RESOURCE_TYPES.COAL: "COAL",
            Constants.RESOURCE_TYPES.URANIUM: "URANIUM",
        }
        fuel_now = {}
        for type, type_upper in type_map.items():
            fuel_now = game.stats["teamStats"][self.team]["resourcesCollected"][type] * game.configs["parameters"]["RESOURCE_TO_FUEL_RATE"][type_upper]
            rewards["rew/r_%s" % type] = (fuel_now - self.fuel_last[type]) / self.fuel_start[type]
            self.stats["game/%s_rate_mined" % type] = fuel_now / self.fuel_start[type]
            self.fuel_last[type] = fuel_now
        
        # Give more incentive for coal and uranium
        rewards["rew/r_%s" % Constants.RESOURCE_TYPES.COAL] *= 2
        rewards["rew/r_%s" % Constants.RESOURCE_TYPES.URANIUM] *= 4
        
        # Give a reward based on amount of fuel collected. 1.0 reward for each 20K fuel gathered.
        fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
        rewards["rew/r_fuel_collected"] = ( (fuel_collected - self.fuel_collected_last) / 20000 )
        self.fuel_collected_last = fuel_collected

        # Give a reward for unit creation/death. 0.05 reward per unit.
        rewards["rew/r_units"] = (unit_count - self.units_last) * 0.05
        self.units_last = unit_count

        # Give a reward for unit creation/death. 0.1 reward per city.
        rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.1
        self.city_tiles_last = city_tile_count

        # Tiny reward for research to help. Up to 0.5 reward for this.
        rewards["rew/r_research"] = (research - self.research_last) / (200 * 2)
        self.research_last = research

#         # Give a reward accoring to the number of city tiles more than your opponent
#         rewards["rew/r_city_tiles_more_opponent"] = (city_tile_count_opponent - city_tile_count) * 0.5

        # Give a reward up to around 50.0 based on number of city tiles at the end of the game
        rewards["rew/r_city_tiles_end"] = 0
        if is_game_finished:
            self.is_last_turn = True
            rewards["rew/r_city_tiles_end"] = city_tile_count

#         # Give a big reward if you win!
#         rewards["rew/r_win"] = 0
#         if is_game_finished and (city_tile_count_opponent > city_tile_count):
#             self.is_last_turn = True
#             rewards["rew/r_win"] = 10.0
        
        # Update the stats and total reward
        reward = 0
        for name, value in rewards.items():
            self.stats[name] += value
            reward += value
        self.stats["rew/r_total"] += reward

        # Print the final game stats sometimes
        if is_game_finished and random.random() <= 0.15:
            stats_string = []
            for key, value in self.stats.items():
                stats_string.append("%s=%.2f" % (key, value))
            print(",".join(stats_string))


        return reward
        
    

    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn. Not used in training, only inference.
        Returns: Array of actions to perform.
        """
        start_time = time.time()
        actions = []
        new_turn = True

        # Inference the model per-unit
        units = game.state["teamStates"][team]["units"].values()
        for unit in units:
            if unit.can_act():
                obs = self.get_observation(game, unit, None, unit.team, new_turn)
                action_code, _states = self.model.predict(obs, deterministic=False)
                if action_code is not None:
                    actions.append(
                        self.action_code_to_action(action_code, game=game, unit=unit, city_tile=None, team=unit.team))
                new_turn = False

        # Inference the model per-city
        unit_count = len(game.state["teamStates"][0]["units"])
        rp = game.state["teamStates"][team]["researchPoints"]
        researched_uranium = False
        city_tile_count = 0
        for tmp_city_tile in game.cities:
            if game.cities[tmp_city_tile].team == team:
                city_tile_count += 1
        cities = game.cities.values()
        for city in cities:
            if city.team == team:
                for cell in city.city_cells:
                    city_tile = cell.city_tile
                    x = city_tile.pos.x
                    y = city_tile.pos.y
                    if rp >= 200:
                        researched_uranium = True
                    if city_tile.can_act():
                        # obs = self.get_observation(game, None, city_tile, city.team, new_turn)
                        # action_code, _states = self.model.predict(obs, deterministic=False)
                        if unit_count < city_tile_count: 
                            actions.append(SpawnWorkerAction(
                                    game=game,
                                    unit_id=None,
                                    unit=unit,
                                    city_id=city_tile.city_id if city_tile else None,
                                    citytile=city_tile,
                                    team=team,
                                    x=x,
                                    y=y
                                ))
                            unit_count += 1
                        elif not researched_uranium:
                            actions.append(ResearchAction(
                                game=game,
                                unit_id=None,
                                unit=unit,
                                city_id=city_tile.city_id if city_tile else None,
                                citytile=city_tile,
                                team=team,
                                x=x,
                                y=y
                            ))
                            rp += 1
                        new_turn = False

        time_taken = time.time() - start_time
        if time_taken > 0.5:  # Warn if larger than 0.5 seconds.
            print("WARNING: Inference took %.3f seconds for computing actions. Limit is 1 second." % time_taken,
                  file=sys.stderr)

        return actions
