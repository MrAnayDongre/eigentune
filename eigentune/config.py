# Copyright 2024 MrAnayDongre
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# eigentune/config.py

from typing import List, Optional

class EigenTuneConfig:
    """
    Configuration class for an EigenTune model.

    This class holds all the configuration parameters for applying EigenTune.

    Args:
        rank (int, optional): The rank of the singular value decomposition update.
            This is the number of top singular values to be fine-tuned.
            Defaults to 4.
        target_modules (list[str] | None, optional): The list of module names
            or substrings to apply EigenTune to (e.g., ["q_proj", "v_proj"]).
            If None, all linear layers are targeted. Defaults to None.
    """
    def __init__(
        self,
        rank: int = 4,
        target_modules: Optional[List[str]] = None,
    ):
        self.rank = rank
        self.target_modules = target_modules