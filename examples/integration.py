import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from leitwerk import Optimizer, Parameter


@dataclass(frozen=True, slots=True)
class Params:
    attack_threshold: Annotated[float, Parameter()]
    worker_limit: Annotated[float, Parameter(mean=66, scale=10, min=12)]


params_file = Path("data/params.json")
opt = Optimizer(Params)

if params_file.exists():
    schema_diff = opt.load(json.loads(params_file.read_text()))

context = {"enemy_race": "Protoss"}
params = opt.ask(context)
print(opt.mean)
