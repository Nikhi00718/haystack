# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.components.generators.chat.types import ChatGenerator

_CHAT_COMPLETIONS_GENERATORS = {"OpenAIChatGenerator", "AzureOpenAIChatGenerator"}
_RESPONSES_GENERATORS = {"OpenAIResponsesChatGenerator", "AzureOpenAIResponsesChatGenerator"}


def _generator_output_token_limit_key(chat_generator: ChatGenerator) -> str | None:
    """Return the output-token parameter used by a known built-in Chat Generator."""
    class_names = {cls.__name__ for cls in type(chat_generator).__mro__}
    if class_names & _RESPONSES_GENERATORS:
        return "max_output_tokens"
    if class_names & _CHAT_COMPLETIONS_GENERATORS:
        return "max_completion_tokens"
    return None


def _resolve_output_token_limit(chat_generator: ChatGenerator, default_limit: int) -> tuple[int, dict[str, Any] | None]:
    """
    Resolve an effective output-token limit and runtime kwargs for a Chat Generator.

    A recognized limit configured directly on a built-in generator wins and is not repeated at runtime. When the
    built-in generator has no configured limit, the default is returned as its provider-specific runtime setting.
    Unknown generators receive no runtime setting because the ChatGenerator protocol does not standardize the key.

    :param chat_generator: The generator whose output should be limited.
    :param default_limit: The positive fallback output-token limit.
    :returns: The effective limit and provider-specific runtime generation kwargs, or None for no runtime kwargs.
    """
    limit_key = _generator_output_token_limit_key(chat_generator=chat_generator)
    if limit_key is None:
        return default_limit, None

    configured = getattr(chat_generator, "generation_kwargs", None)
    if isinstance(configured, dict) and limit_key in configured:
        value = configured[limit_key]
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value, None
        # The generator owns this setting. Do not silently replace an invalid value; let it report the problem.
        return default_limit, None
    return default_limit, {limit_key: default_limit}
