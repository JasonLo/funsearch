# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from collections.abc import Sequence

import numpy as np

from funsearch import evaluator
from funsearch import programs_database
from funsearch import llm


class Sampler:
    """Node that samples program continuations and sends them for analysis."""

    def __init__(
        self,
        database: programs_database.ProgramsDatabase,
        evaluators: Sequence[evaluator.Evaluator],
        model: llm.LLM,
    ) -> None:
        self._database = database
        self._evaluators = evaluators
        self._llm = model

    def sample(self):
        """Continuously gets prompts, samples programs, sends them for analysis."""
        prompt = self._database.get_prompt()
        samples = self._llm.draw_samples(prompt.code)
        # This loop can be executed in parallel on remote evaluator machines.
        for sample in samples:
            chosen_evaluator = np.random.choice(self._evaluators)
            chosen_evaluator.analyse(sample, prompt.island_id, prompt.version_generated)
