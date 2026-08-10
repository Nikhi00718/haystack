# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.hooks.compaction import SummarizationCompactor
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY
from test.hooks.compaction.helpers import FakeCounter, tool_call, tool_result

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")
COUNTER = FakeCounter(chars_per_token=1)


def recording_generator(responses: list[str | Exception]) -> tuple[MockChatGenerator, list[dict[str, Any]]]:
    """Build a MockChatGenerator whose response function records prompts and can raise queued failures."""
    queued = list(responses)
    calls: list[dict[str, Any]] = []

    def respond(messages: list[ChatMessage]) -> str:
        calls.append({"messages": messages})
        response = queued.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    return MockChatGenerator(response_fn=respond), calls


def summary(text: str, source: str) -> ChatMessage:
    return ChatMessage.from_user(
        f"<conversation_summary>\n{text}\n</conversation_summary>",
        meta={_COMPACTION_META_KEY: {"strategy": "summarization", "source": source}},
    )


def transcript(call: dict[str, Any]) -> str:
    return call["messages"][-1].text


class TestSummarizationCompactor:
    def test_summarizes_the_minimum_number_of_oldest_historical_turns(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("oldest question " * 30),
            ChatMessage.from_assistant("oldest answer " * 30),
            ChatMessage.from_user("recent question"),
            ChatMessage.from_assistant("recent answer"),
            ChatMessage.from_user("current task"),
            ChatMessage.from_assistant("current step"),
        ]
        generator, calls = recording_generator(["short historical summary"])
        retained_without_oldest = [messages[0], *messages[3:]]
        target = COUNTER.count(retained_without_oldest) + 100
        compacted = SummarizationCompactor(generator, max_summary_tokens=100).compact(
            messages=messages, target_tokens=target, token_counter=COUNTER
        )

        assert compacted is not None
        assert len(calls) == 1
        assert "oldest question" in transcript(calls[0])
        assert "recent question" not in transcript(calls[0])
        assert compacted[0] == messages[0]
        assert compacted[2:] == messages[3:]
        assert messages == [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("oldest question " * 30),
            ChatMessage.from_assistant("oldest answer " * 30),
            ChatMessage.from_user("recent question"),
            ChatMessage.from_assistant("recent answer"),
            ChatMessage.from_user("current task"),
            ChatMessage.from_assistant("current step"),
        ]

    def test_reaches_current_steps_in_the_same_call_after_historical_context(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("old question " * 40),
            ChatMessage.from_assistant("old answer " * 40),
            ChatMessage.from_user("current task"),
            tool_call("old"),
            tool_result("old result " * 40, call_id="old"),
            tool_call("new"),
            tool_result("new result", call_id="new"),
        ]
        generator, calls = recording_generator(["history", "old step"])
        compacted = SummarizationCompactor(generator, max_summary_tokens=1).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )

        assert compacted is not None
        assert len(calls) == 2
        assert "old question" in transcript(calls[0])
        assert "old result" in transcript(calls[1])
        assert compacted[-2:] == messages[-2:]
        assert [m.meta[_COMPACTION_META_KEY]["source"] for m in compacted if _COMPACTION_META_KEY in m.meta] == [
            "historical_turns",
            "current_task_steps",
        ]

    def test_consolidates_historical_summaries_before_current_steps(self):
        messages = [
            ChatMessage.from_system("rules"),
            summary("first history " * 20, "historical_turns"),
            summary("second history " * 20, "historical_turns"),
            ChatMessage.from_user("current task"),
            tool_call("old"),
            tool_result("old result " * 30, call_id="old"),
            tool_call("new"),
            tool_result("new result", call_id="new"),
        ]
        generator, calls = recording_generator(["combined history", "old step"])
        compacted = SummarizationCompactor(generator, max_summary_tokens=1).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )

        assert compacted is not None
        assert len(calls) == 2
        assert "first history" in transcript(calls[0])
        assert "old result" not in transcript(calls[0])
        assert "old result" in transcript(calls[1])

    def test_consolidates_current_summaries_before_more_raw_steps(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("current task"),
            summary("first step summary " * 20, "current_task_steps"),
            summary("second step summary " * 20, "current_task_steps"),
            tool_call("old"),
            tool_result("old result " * 30, call_id="old"),
            tool_call("new"),
            tool_result("new result", call_id="new"),
        ]
        generator, calls = recording_generator(["combined steps", "old step"])
        compacted = SummarizationCompactor(generator, max_summary_tokens=1).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )

        assert compacted is not None
        assert len(calls) == 2
        assert "first step summary" in transcript(calls[0])
        assert "old result" in transcript(calls[1])
        assert compacted[-2:] == messages[-2:]

    def test_returns_partial_progress_when_a_later_tier_fails_by_default(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("old question " * 30),
            ChatMessage.from_assistant("old answer " * 30),
            ChatMessage.from_user("current task"),
            ChatMessage.from_assistant("old step " * 30),
            ChatMessage.from_assistant("new step"),
        ]
        generator, calls = recording_generator(["history", RuntimeError("provider unavailable")])
        compacted = SummarizationCompactor(generator, max_summary_tokens=1).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )

        assert compacted is not None
        assert len(calls) == 2
        assert any(
            message.meta.get(_COMPACTION_META_KEY, {}).get("source") == "historical_turns" for message in compacted
        )
        assert messages[-2:] == compacted[-2:]

    def test_raises_when_configured_and_summary_does_not_shrink_context(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("old"),
            ChatMessage.from_assistant("answer"),
            ChatMessage.from_user("current"),
        ]
        generator, _ = recording_generator(["much longer summary " * 100])
        with pytest.raises(RuntimeError, match="did not reduce"):
            SummarizationCompactor(generator, max_summary_tokens=1, raise_on_failure=True).compact(messages, 1, COUNTER)

    def test_returns_none_without_calling_generator_when_context_fits(self):
        generator, calls = recording_generator(["unused"])
        messages = [ChatMessage.from_system("rules"), ChatMessage.from_user("task")]
        assert (
            SummarizationCompactor(generator).compact(messages=messages, target_tokens=10_000, token_counter=COUNTER)
            is None
        )
        assert calls == []

    def test_summary_is_a_user_message_with_compaction_metadata(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("old " * 100),
            ChatMessage.from_assistant("answer " * 100),
            ChatMessage.from_user("task"),
        ]
        compacted = SummarizationCompactor(MockChatGenerator("summary"), max_summary_tokens=1).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )
        assert compacted is not None
        generated = compacted[1]
        assert generated.is_from(ChatRole.USER)
        assert generated.meta[_COMPACTION_META_KEY] == {
            "strategy": "summarization",
            "summarized_messages": 2,
            "source": "historical_turns",
        }

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"min_keep_steps": -1}, "`min_keep_steps` must be at least 0"),
            ({"max_summary_tokens": 0}, "`max_summary_tokens` must be a positive"),
        ],
    )
    def test_rejects_invalid_settings(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SummarizationCompactor(MockChatGenerator("summary"), **kwargs)

    def test_serde_round_trip(self):
        compactor = SummarizationCompactor(
            MockChatGenerator("summary"),
            min_keep_steps=2,
            max_summary_tokens=321,
            summary_instruction="custom",
            raise_on_failure=True,
        )
        restored = SummarizationCompactor.from_dict(compactor.to_dict())
        assert isinstance(restored.chat_generator, MockChatGenerator)
        assert restored.min_keep_steps == 2
        assert restored.max_summary_tokens == 321
        assert restored.summary_instruction == "custom"
        assert restored.raise_on_failure is True


@pytest.mark.asyncio
async def test_async_compaction_uses_async_generator():
    messages = [
        ChatMessage.from_system("rules"),
        ChatMessage.from_user("old " * 100),
        ChatMessage.from_assistant("answer " * 100),
        ChatMessage.from_user("task"),
    ]
    generator, calls = recording_generator(["async summary"])
    compacted = await SummarizationCompactor(generator, max_summary_tokens=1).compact_async(
        messages=messages, target_tokens=1, token_counter=COUNTER
    )
    assert compacted is not None
    assert len(calls) == 1
