import json

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from loguru import logger
from sc2 import maps
from sc2.main import run_game
from sc2.player import Bot, Computer

from leitwerk import Optimizer, Parameter
from sc2.bot_ai import BotAI
from sc2.data import Result, Race, Difficulty


DATA_PATH = Path("./data")
PARAMS_FILE = DATA_PATH / "params.json"


@dataclass
class BotParams:
    attack_threshold: Annotated[float, Parameter(min=0, max=1)]
    retreat_threshold: Annotated[float, Parameter(min=0, max=1)]


class LearningBot(BotAI):

    async def on_start(self):
        # restore state from disk
        self.optimizer = Optimizer(BotParams)
        DATA_PATH.mkdir(exist_ok=True)
        if PARAMS_FILE.exists():
            with PARAMS_FILE.open() as f:
                state = json.load(f)
            diff = self.optimizer.load(state)
            logger.info(diff)
        context = self.enemy_race.name  # optional: matchup-based mirror sampling
        self.params = self.optimizer.ask(context)
        logger.info(self.params)

    async def on_step(self, iteration):
        mineral_patch = self.mineral_field.closest_to(self.start_location)
        for worker in self.workers:
            if worker.shield_percentage > self.params.attack_threshold:
                if self.enemy_structures:
                    worker.attack(self.enemy_structures.random.position)
                else:
                    worker.attack(self.enemy_start_locations[0])
            elif worker.shield_health_percentage < self.params.retreat_threshold:
                worker.gather(mineral_patch)
        if self.supply_used == 0:
            await self.client.debug_kill_unit(self.structures)

    async def on_end(self, game_result: Result) -> None:
        # primary objective: win
        win_loss = {
            Result.Victory: +1,
            Result.Tie: 0,
            Result.Defeat: -1,
        }[game_result]
        # secondary objective: be cost-effective
        efficiency = self.state.score.killed_value_units / max(1, self.state.score.lost_minerals_economy)
        score = (win_loss, efficiency)
        logger.info(score)
        tell_result = self.optimizer.tell(score)
        logger.info(tell_result)
        state = self.optimizer.save()
        with PARAMS_FILE.open("w") as f:
            json.dump(state, f, indent=2)


def main():
    while True:
        run_game(
            maps.get("TorchesAIE_v4"),
            [Bot(Race.Protoss, LearningBot()), Computer(Race.Protoss, Difficulty.CheatInsane)],
            realtime=False,
        )


if __name__ == "__main__":
    main()
