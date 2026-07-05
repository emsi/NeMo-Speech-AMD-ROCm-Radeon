# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""NeMo services for Pipecat voice-agent pipelines."""

from importlib import import_module

__all__ = [
    "HuggingFaceLLMService",
    "NemoDiarService",
    "NemoSTTService",
    "NeMoFastPitchHiFiGANTTSService",
    "NeMoTurnTakingService",
]

_SERVICE_MODULES = {
    "HuggingFaceLLMService": ".llm",
    "NemoDiarService": ".diar",
    "NemoSTTService": ".stt",
    "NeMoFastPitchHiFiGANTTSService": ".tts",
    "NeMoTurnTakingService": ".turn_taking",
}


def __getattr__(name: str):
    if name not in _SERVICE_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_SERVICE_MODULES[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
