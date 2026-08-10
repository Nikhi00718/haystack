# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.components.generators.chat import MockChatGenerator
from haystack.components.generators.chat.utils import _generator_output_token_limit_key, _resolve_output_token_limit


class OpenAIChatGenerator(MockChatGenerator):
    def __init__(self, generation_kwargs=None):
        super().__init__("response")
        self.generation_kwargs = generation_kwargs or {}


class OpenAIResponsesChatGenerator(MockChatGenerator):
    def __init__(self, generation_kwargs=None):
        super().__init__("response")
        self.generation_kwargs = generation_kwargs or {}


class CustomGenerator(MockChatGenerator):
    generation_kwargs = {"max_tokens": 7}


def test_resolves_chat_completions_limit():
    generator = OpenAIChatGenerator()
    assert _generator_output_token_limit_key(generator) == "max_completion_tokens"
    assert _resolve_output_token_limit(generator, 100) == (100, {"max_completion_tokens": 100})


def test_resolves_responses_limit():
    generator = OpenAIResponsesChatGenerator()
    assert _generator_output_token_limit_key(generator) == "max_output_tokens"
    assert _resolve_output_token_limit(generator, 100) == (100, {"max_output_tokens": 100})


def test_configured_generator_limit_wins_without_mutation():
    generator = OpenAIChatGenerator({"temperature": 0, "max_completion_tokens": 23})
    original = dict(generator.generation_kwargs)
    assert _resolve_output_token_limit(generator, 100) == (23, None)
    assert generator.generation_kwargs == original


def test_unknown_generator_receives_no_guessed_runtime_setting():
    assert _generator_output_token_limit_key(CustomGenerator()) is None
    assert _resolve_output_token_limit(CustomGenerator(), 100) == (100, None)
