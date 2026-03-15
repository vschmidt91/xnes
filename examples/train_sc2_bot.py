import json
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Annotated, SupportsFloat, cast

import matplotlib.pyplot as plt
import numpy as np
from leitwerk import Optimizer, Parameter
from loguru import logger
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race, Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.position import Point2
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data"
PARAMS_FILE = DATA_PATH / "params.json"
HISTORY_FILE = DATA_PATH / "history.json"
PLOT_FILE = DATA_PATH / "plot.png"
MAP_FILE = ROOT / "resources" / "PylonAIE_v4.SC2Map"
MOVING_AVERAGE_WINDOW = 10


@dataclass
class BotParams:
    max_group_distance: Annotated[float, Parameter(loc=3, scale=3, min=0)]
    shield_threshold: Annotated[float, Parameter(min=0, max=1)]


class LearningBot(BotAI):
    optimizer = Optimizer(BotParams)
    params: BotParams | None = None

    async def on_start(self) -> None:
        self.townhalls[0].train(UnitTypeId.PROBE)
        DATA_PATH.mkdir(exist_ok=True)
        if PARAMS_FILE.exists():
            with PARAMS_FILE.open() as f:
                state = json.load(f)
            diff = self.optimizer.load(state)
            logger.info(diff)

        # sammple now if enemy race is known, wait for vision otherwise
        if self.enemy_race in {Race.Protoss, Race.Terran, Race.Zerg}:
            self.sample_params(self.enemy_race)

    def sample_params(self, enemy_race: Race) -> None:
        if self.params is not None:
            # multiple ask calls are not supported
            return
        context = {
            "enemy_race": enemy_race.name,
        }
        logger.info(f"{context=}")
        self.params = self.optimizer.ask(context)
        logger.info(f"{self.params=}")

    async def on_step(self, iteration: int) -> None:

        # delayed sampling
        if self.params is None:
            if self.all_enemy_units:
                self.sample_params(self.all_enemy_units.first.race)
            else:
                for worker in self.workers:
                    worker.attack(self.enemy_start_locations[0])
            return

        mineral_patch = self.mineral_field.closest_to(self.start_location)
        group_center = _medoid([u.position for u in self.workers]) if self.workers else self.start_location
        for worker in self.workers:
            if worker.distance_to(group_center) > self.params.max_group_distance:
                worker.move(group_center)
            elif worker.shield_percentage < self.params.shield_threshold:
                worker.gather(mineral_patch)
            elif self.enemy_structures:
                worker.attack(self.enemy_structures.random.position)
            else:
                worker.attack(self.enemy_start_locations[0])

        # resign for speedup
        if self.supply_used == 0:
            await self.client.chat_send("gg", False)
            await self.client.leave()

    async def on_end(self, game_result: Result) -> None:
        outcome = {
            Result.Victory: 1.0,
            Result.Tie: 0.5,
            Result.Defeat: 0.0,
        }[game_result]

        efficiency = np.log1p(self.state.score.killed_value_units) - np.log1p(self.state.score.lost_minerals_economy)
        result = (outcome, efficiency)
        logger.info(f"{result=}")
        tell_result = self.optimizer.tell(result)
        logger.info(tell_result)
        history = load_history()
        history.append({"outcome": outcome, "efficiency": efficiency, **flatten_numeric_fields(self.params)})
        save_history(history)
        save_plot(history)
        state = self.optimizer.save()
        with PARAMS_FILE.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)


def _medoid(points: Sequence[Point2]) -> Point2:
    distances = _pairwise_distances(points)
    medoid_index = distances.sum(axis=1).argmin()
    return points[medoid_index]


def _pairwise_distances(positions: Sequence[Point2]) -> np.ndarray:
    return squareform(pdist(positions), checks=False)


def load_history() -> list[dict[str, float]]:
    DATA_PATH.mkdir(exist_ok=True)
    if not HISTORY_FILE.exists():
        return []
    with HISTORY_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def save_history(history: list[dict[str, float]]) -> None:
    DATA_PATH.mkdir(exist_ok=True)
    with HISTORY_FILE.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def moving_average(values: list[float], window: int = MOVING_AVERAGE_WINDOW) -> np.ndarray:
    if not values:
        return np.array([], dtype=float)
    series = np.asarray(values, dtype=float)
    cumulative = np.cumsum(series)
    trailing = cumulative.copy()
    trailing[window:] -= cumulative[:-window]
    counts = np.minimum(np.arange(1, len(series) + 1), window)
    return trailing / counts


def flatten_numeric_fields(value: object, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if is_dataclass(value):
        for field in fields(value):
            key = f"{prefix}.{field.name}" if prefix else field.name
            result.update(flatten_numeric_fields(getattr(value, field.name), key))
        return result
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric_fields(item, name))
        return result
    return {prefix: float(cast(SupportsFloat, value))}


def metric_names(history: list[dict[str, float]]) -> list[str]:
    names = sorted({key for entry in history for key in entry})
    preferred = ["outcome", "efficiency"]
    return [name for name in preferred if name in names] + [name for name in names if name not in preferred]


def plot_metric(
    ax: plt.Axes,
    history: list[dict[str, float]],
    metric: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    values = np.array([entry.get(metric, np.nan) for entry in history], dtype=float)
    games = np.arange(1, len(values) + 1)
    valid = ~np.isnan(values)
    ax.scatter(games[valid], values[valid], s=18, alpha=0.7)
    if np.any(valid):
        ax.plot(games[valid], moving_average(values[valid].tolist()), linewidth=2)
    ax.set(ylabel=ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)


def save_plot(history: list[dict[str, float]]) -> None:
    DATA_PATH.mkdir(exist_ok=True)
    metrics = metric_names(history)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics, strict=True):
        plot_metric(ax, history, metric, metric, ylim=(0, 1) if metric == "outcome" else None)
        ax.set_title(metric)
    axes[-1].set_xlabel("game")
    fig.tight_layout()
    fig.savefig(PLOT_FILE)
    plt.close(fig)


def main() -> None:
    while True:
        run_game(
            maps.Map(MAP_FILE),
            [Bot(Race.Protoss, LearningBot()), Computer(Race.Random, Difficulty.CheatInsane)],
            realtime=False,
        )


if __name__ == "__main__":
    main()
