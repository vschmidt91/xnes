from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from leitwerk import OptimizerSession, Parameter
from loguru import logger
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race, Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer
from utils import flatten_numeric_fields, load_history, save_history, save_plot, simulate_combat

EXAMPLES_DIR = Path(__file__).resolve().parent
PARAMS_FILE = EXAMPLES_DIR / "data" / "params.json"
MAP_FILE = EXAMPLES_DIR / "maps" / "PylonAIE_v4.SC2Map"

RACE = Race.Protoss
ENEMY_RACE = Race.Protoss
ENEMY_DIFFICULTY = Difficulty.VeryHard


@dataclass
class BotParams:
    simulation_time: Annotated[float, Parameter(mean=1, min=0)]
    retreat_threshold: Annotated[float, Parameter()]


class LearningBot(BotAI):
    def __init__(self) -> None:
        super().__init__()
        self.optimizer = OptimizerSession(PARAMS_FILE, BotParams)
        self.params = self.optimizer.ask()

    async def on_start(self) -> None:
        self.townhalls[0].train(UnitTypeId.PROBE)
        if self.optimizer.restored:
            logger.info("Restored optimizer from file")
        else:
            logger.info("Initialized optimizer state")
        logger.info(self.optimizer.schema_diff)

    async def on_step(self, iteration: int) -> None:

        # early resign for faster training
        if self.supply_used == 0:
            logger.info("Resigning")
            await self.client.leave()
            return

        mineral_patch = self.mineral_field.closest_to(self.start_location)
        simulation = simulate_combat(self.workers | self.enemy_units, self.params.simulation_time)

        for worker in self.workers:
            if simulation[worker] < self.params.retreat_threshold:
                worker.gather(mineral_patch)
            elif self.enemy_structures:
                worker.attack(self.enemy_structures.random.position)
            else:
                worker.attack(self.enemy_start_locations[0])

    async def on_end(self, game_result: Result) -> None:
        logger.info(f"Game ended at {self.time_formatted} with outcome {game_result.name}")
        outcome = {Result.Victory: 1.0, Result.Tie: 0.5, Result.Defeat: 0.0}[game_result]
        efficiency = self.state.score.killed_value_units / self.time
        result = outcome, efficiency
        logger.info(f"Encoded objective value: {result}")
        report = self.optimizer.tell(result)
        logger.info(f"Optimizer tell report: {report}")
        output_dir = EXAMPLES_DIR / "data"
        history = load_history(output_dir)
        history.append({"result": list(result), **flatten_numeric_fields(self.params)})
        save_history(history, output_dir)
        save_plot(history, output_dir, ["outcome", "efficiency"])
        logger.info(f"Stored results and plots using {len(history)} samples")


if __name__ == "__main__":
    while True:
        bot = Bot(RACE, LearningBot())
        opponent = Computer(ENEMY_RACE, ENEMY_DIFFICULTY)
        run_game(maps.Map(MAP_FILE), [bot, opponent], realtime=False)
