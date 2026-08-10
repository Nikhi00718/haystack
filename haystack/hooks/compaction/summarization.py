# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import logging
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.generators.chat.utils import _resolve_output_token_limit
from haystack.core.serialization import component_to_dict, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction.types import Compactor
from haystack.hooks.compaction.utils import (
    _COMPACTION_META_KEY,
    _current_step_groups,
    _historical_turn_groups,
    _is_compaction_message,
    _latest_user_index,
    _leading_system_end,
    _messages_at,
    _messages_except,
)
from haystack.token_counters import TokenCounter
from haystack.token_counters.utils import _rendered_conversation
from haystack.utils.async_utils import _execute_component_async
from haystack.utils.deserialization import deserialize_component_inplace
from haystack.utils.experimental import _experimental

logger = logging.getLogger(__name__)

# Recorded as the strategy on every summary this compactor produces, so a later run can recognize its own summaries.
_STRATEGY = "summarization"

# Recorded as the `source` on a summary, naming the stretch of conversation it stands in for. Compaction gives these up
# in order, so the Agent's current task is the last thing to go.
_HISTORICAL_TURNS = "historical_turns"
_HISTORICAL_SUMMARIES = "historical_summaries"
_CURRENT_TASK_SUMMARIES = "current_task_summaries"
_CURRENT_TASK_STEPS = "current_task_steps"

_DEFAULT_SUMMARY_INSTRUCTION = """You are compacting part of a conversation between a user and an AI agent so the \
agent can keep working with fewer tokens. Write a self-contained summary that preserves:
- The user's goal, requirements, constraints, and preferences.
- Decisions and the reasoning behind them.
- Work already completed and important tool results.
- Exact file paths, URLs, identifiers, and references to stored data.
- Unresolved work and the immediate next step.

Fold any existing <conversation_summary> blocks into one summary. Record only what the conversation shows. Do not \
infer or add advice. Use plain prose or short bullets, and do not address the user."""


def _is_summary(message: ChatMessage) -> bool:
    """Whether a message is a summary this strategy wrote."""
    return _is_compaction_message(message=message, strategy=_STRATEGY)


def _summary_indices(messages: list[ChatMessage], start: int, end: int) -> list[int]:
    """Return the positions of this strategy's summaries within a bounded part of a conversation."""
    return [index for index in range(start, end) if _is_summary(message=messages[index])]


def _summarizable_turn_groups(messages: list[ChatMessage], system_end: int, task_index: int | None) -> list[list[int]]:
    """
    Return the historical turns that still hold raw conversation, oldest turn first.

    Summaries this strategy already wrote are left out of their turn, so summarizing the turn folds that summary into
    the summary this run produces. A turn that is nothing but summaries has nothing left to give up and is dropped.
    """
    groups = [
        [index for index in group if not _is_summary(message=messages[index])]
        for group in _historical_turn_groups(messages=messages, system_end=system_end, task_index=task_index)
    ]
    return [group for group in groups if group]


def _groups_to_summarize(
    messages: list[ChatMessage],
    groups: list[list[int]],
    target_tokens: int,
    summary_tokens: int,
    token_counter: TokenCounter,
) -> list[int]:
    """
    Return the fewest oldest groups whose removal makes room for a summary of the expected size.

    Groups are taken oldest first and counting stops as soon as what remains, plus the summary that replaces them,
    fits the target. When even taking all of them is not enough, all of them are returned.
    """
    selected: list[int] = []
    for group in groups:
        selected.extend(group)
        remaining = token_counter.count(messages=_messages_except(messages=messages, indices=selected))
        if remaining + summary_tokens <= target_tokens:
            break
    return selected


def _summary_message(text: str, summarized_messages: int, source: str) -> ChatMessage:
    """Build the marked user message that stands in for the messages the summary replaced."""
    body = f"<conversation_summary>\n{text.strip()}\n</conversation_summary>"
    meta = {_COMPACTION_META_KEY: {"strategy": _STRATEGY, "summarized_messages": summarized_messages, "source": source}}
    return ChatMessage.from_user(text=body, meta=meta)


def _replace_indices(messages: list[ChatMessage], indices: list[int], summary: ChatMessage) -> list[ChatMessage]:
    """Replace the selected messages, which need not be contiguous, with one summary at the oldest one's position."""
    selected = set(indices)
    insertion_index = min(indices)
    compacted: list[ChatMessage] = []
    for index, message in enumerate(messages):
        if index == insertion_index:
            compacted.append(summary)
        if index not in selected:
            compacted.append(message)
    return compacted


@_experimental
class SummarizationCompactor(Compactor):
    """
    Condenses old historical turns first, then old steps from the Agent's current task.

    Leading system messages and the latest real user message are always kept. Historical turns are summarized in full,
    oldest first. Summaries normally accumulate so they are not repeatedly rewritten; if every historical turn has
    already been summarized and more space is needed, those historical summaries are folded into one before any
    current-task steps are summarized. An assistant message and all immediately following tool results form one step,
    so tool calls are never separated from their results.

    Each summary is requested within `max_summary_tokens`. Built-in OpenAI Chat Completions and Responses generators
    receive the corresponding runtime output limit unless they already configure one in their `generation_kwargs`, in
    which case the generator's setting wins. Other generators receive the same limit as prompt guidance and the actual
    result is measured before it is accepted.

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator
    from haystack.hooks.compaction import CompactionHook, SummarizationCompactor

    summary_generator = OpenAIResponsesChatGenerator(model="gpt-5.4-nano")
    hook = CompactionHook(
        compactor=SummarizationCompactor(chat_generator=summary_generator),
        context_window=400_000,
        compact_at=0.7,
        compact_to=0.4,
    )
    agent = Agent(chat_generator=agent_generator, tools=[web_search], hooks={"before_llm": [hook]})
    ```
    """

    def __init__(
        self,
        chat_generator: ChatGenerator,
        *,
        min_keep_steps: int = 1,
        max_summary_tokens: int = 1024,
        summary_instruction: str | None = None,
        raise_on_failure: bool = False,
    ) -> None:
        """
        Initialize the compactor.

        :param chat_generator: The Chat Generator used to write summaries. Its configured output-token limit takes
            precedence over `max_summary_tokens` when the generator exposes a recognized setting.
        :param min_keep_steps: The fewest complete recent Agent steps to keep, even when they exceed the target.
        :param max_summary_tokens: The output-token budget reserved for each summary. Known built-in generators receive
            the corresponding runtime generation setting unless one is already configured on the generator.
        :param summary_instruction: An instruction replacing the built-in summary prompt.
        :param raise_on_failure: Whether a failed or non-shrinking summarization raises. By default the failure is
            logged and any successful partial compaction is returned.
        :raises ValueError: If `min_keep_steps` is negative or `max_summary_tokens` is not positive.
        """
        if min_keep_steps < 0:
            raise ValueError(f"`min_keep_steps` must be at least 0, got {min_keep_steps}.")
        if max_summary_tokens < 1:
            raise ValueError(f"`max_summary_tokens` must be a positive number of tokens, got {max_summary_tokens}.")
        self.chat_generator = chat_generator
        self.min_keep_steps = min_keep_steps
        self.max_summary_tokens = max_summary_tokens
        self.summary_instruction = summary_instruction or _DEFAULT_SUMMARY_INSTRUCTION
        self.raise_on_failure = raise_on_failure

    def compact(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        """
        Return a progressively summarized conversation, or None when no useful reduction is possible.

        :param messages: The conversation to compact, ordered oldest to newest.
        :param target_tokens: The token budget the compacted messages should aim to fit.
        :param token_counter: The counter used both to plan compaction and verify generated summaries.
        :returns: A smaller replacement conversation, or None when nothing was reduced.
        """
        summary_tokens, run_kwargs = self._summary_limit()
        working = list(messages)
        while True:
            plan = self._next_summary(
                messages=working,
                target_tokens=target_tokens,
                token_counter=token_counter,
                summary_tokens=summary_tokens,
            )
            if plan is None:
                break
            indices, source = plan
            prompt = self._prompt(messages=working, indices=indices, summary_tokens=summary_tokens)
            try:
                result = self.chat_generator.run(messages=prompt, **run_kwargs)
                working = self._apply_summary(
                    messages=working, indices=indices, source=source, result=result, token_counter=token_counter
                )
            except Exception as error:
                self._report_failure(error=error)
                break
        return self._reduced(original=messages, working=working, token_counter=token_counter)

    async def compact_async(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        """
        Asynchronously return a progressively summarized conversation.

        :param messages: The conversation to compact, ordered oldest to newest.
        :param target_tokens: The token budget the compacted messages should aim to fit.
        :param token_counter: The counter used both to plan compaction and verify generated summaries.
        :returns: A smaller replacement conversation, or None when nothing was reduced.
        """
        summary_tokens, run_kwargs = self._summary_limit()
        working = list(messages)
        while True:
            plan = self._next_summary(
                messages=working,
                target_tokens=target_tokens,
                token_counter=token_counter,
                summary_tokens=summary_tokens,
            )
            if plan is None:
                break
            indices, source = plan
            prompt = self._prompt(messages=working, indices=indices, summary_tokens=summary_tokens)
            try:
                result = await _execute_component_async(
                    component_instance=self.chat_generator, messages=prompt, **run_kwargs
                )
                working = self._apply_summary(
                    messages=working, indices=indices, source=source, result=result, token_counter=token_counter
                )
            except Exception as error:
                self._report_failure(error=error)
                break
        return self._reduced(original=messages, working=working, token_counter=token_counter)

    def _next_summary(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter, summary_tokens: int
    ) -> tuple[list[int], str] | None:
        """
        Choose the next stretch of conversation to replace with a summary.

        Four tiers are tried in order, so the oldest and least useful context goes first and the Agent's current task
        is given up last:

        1. `_HISTORICAL_TURNS`: the fewest oldest raw turns that make room for a summary.
        2. `_HISTORICAL_SUMMARIES`: nothing raw is left in history, so fold its summaries into one.
        3. `_CURRENT_TASK_SUMMARIES`: fold the summaries earlier steps left behind before giving up more steps.
        4. `_CURRENT_TASK_STEPS`: the fewest oldest steps of the current task, keeping `min_keep_steps` of the newest.

        :param messages: The conversation as it stands, ordered oldest to newest.
        :param target_tokens: The token budget the conversation should come in under.
        :param token_counter: The counter used to measure candidate selections.
        :param summary_tokens: The size a summary is expected to take, reserved when choosing how much to replace.
        :returns: The message indices to summarize and the `source` to record on the resulting summary, or None when
            the conversation already fits or nothing is left that may be given up.
        """
        if token_counter.count(messages=messages) <= target_tokens:
            return None

        # The landmarks everything is measured against: the Agent's instructions, and the user message anchoring the
        # current task. History runs from the instructions up to that anchor, the current task from the anchor on.
        system_end = _leading_system_end(messages=messages)
        task_index = _latest_user_index(messages=messages)
        history_end = task_index if task_index is not None else system_end
        task_start = task_index + 1 if task_index is not None else system_end

        turns = _summarizable_turn_groups(messages=messages, system_end=system_end, task_index=task_index)
        if turns:
            oldest_turns = _groups_to_summarize(
                messages=messages,
                groups=turns,
                target_tokens=target_tokens,
                summary_tokens=summary_tokens,
                token_counter=token_counter,
            )
            return oldest_turns, _HISTORICAL_TURNS

        history_summaries = _summary_indices(messages=messages, start=system_end, end=history_end)
        if len(history_summaries) > 1:
            return history_summaries, _HISTORICAL_SUMMARIES

        # Only steps older than the `min_keep_steps` most recent ones may be given up.
        steps = _current_step_groups(messages=messages, system_end=system_end, task_index=task_index)
        eligible = steps[: max(len(steps) - self.min_keep_steps, 0)]
        if not eligible:
            return None

        task_summaries = _summary_indices(messages=messages, start=task_start, end=len(messages))
        if len(task_summaries) > 1:
            return task_summaries, _CURRENT_TASK_SUMMARIES

        oldest_steps = _groups_to_summarize(
            messages=messages,
            groups=eligible,
            target_tokens=target_tokens,
            summary_tokens=summary_tokens,
            token_counter=token_counter,
        )
        return oldest_steps, _CURRENT_TASK_STEPS

    def _summary_limit(self) -> tuple[int, dict[str, Any]]:
        """Return the token budget for one summary and the run kwargs, if any, that ask the generator to honor it."""
        summary_tokens, generation_kwargs = _resolve_output_token_limit(
            chat_generator=self.chat_generator, default_limit=self.max_summary_tokens
        )
        return summary_tokens, {"generation_kwargs": generation_kwargs} if generation_kwargs else {}

    def _prompt(self, messages: list[ChatMessage], indices: list[int], summary_tokens: int) -> list[ChatMessage]:
        """Build the bounded summarization instruction and the rendered transcript of the selected messages."""
        transcript = _rendered_conversation(_messages_at(messages=messages, indices=indices))
        instruction = (
            f"{self.summary_instruction}\n\nWrite a complete summary in no more than approximately "
            f"{summary_tokens} tokens. Prioritize completeness within that limit so the response is not cut off."
        )
        return [
            ChatMessage.from_system(text=instruction),
            ChatMessage.from_user(text=f"<conversation_to_summarize>\n{transcript}\n</conversation_to_summarize>"),
        ]

    @staticmethod
    def _apply_summary(
        messages: list[ChatMessage],
        indices: list[int],
        source: str,
        result: dict[str, Any],
        token_counter: TokenCounter,
    ) -> list[ChatMessage]:
        """
        Swap the selected messages for the generated summary.

        :raises RuntimeError: If the generator returned no usable text, or if the swap did not make the conversation
            smaller, in which case keeping the raw messages is the better outcome.
        """
        replies = result.get("replies") or []
        text = replies[-1].text if replies else None
        if not text or not text.strip():
            raise RuntimeError("The Chat Generator returned no text to use as a conversation summary.")
        summary = _summary_message(text=text, summarized_messages=len(indices), source=source)
        compacted = _replace_indices(messages=messages, indices=indices, summary=summary)
        before = token_counter.count(messages=messages)
        after = token_counter.count(messages=compacted)
        if after >= before:
            raise RuntimeError(
                f"The generated summary did not reduce the conversation size ({before} tokens before and {after} "
                "tokens after)."
            )
        return compacted

    def _report_failure(self, error: Exception) -> None:
        """Re-raise a failed summarization or log it, so whatever compacted successfully so far is still returned."""
        if self.raise_on_failure:
            raise error
        logger.warning(
            "Summarizing the conversation for context compaction failed; keeping the last successful result. "
            "Error: {error}",
            error=error,
        )

    @staticmethod
    def _reduced(
        original: list[ChatMessage], working: list[ChatMessage], token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        """Return partial or complete progress only when it made the original conversation smaller."""
        if token_counter.count(messages=working) < token_counter.count(messages=original):
            return working
        return None

    def warm_up(self) -> None:
        """Warm up the Chat Generator that writes summaries."""
        if hasattr(self.chat_generator, "warm_up"):
            self.chat_generator.warm_up()

    async def warm_up_async(self) -> None:
        """Warm up the Chat Generator on the serving event loop."""
        warm_up_async = getattr(self.chat_generator, "warm_up_async", None)
        if warm_up_async is not None:
            await warm_up_async()
        elif hasattr(self.chat_generator, "warm_up"):
            self.chat_generator.warm_up()

    def close(self) -> None:
        """Release the Chat Generator's resources."""
        if hasattr(self.chat_generator, "close"):
            self.chat_generator.close()

    async def close_async(self) -> None:
        """Release the Chat Generator's resources."""
        close_async = getattr(self.chat_generator, "close_async", None)
        if close_async is not None:
            await close_async()
        elif hasattr(self.chat_generator, "close"):
            self.chat_generator.close()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the compactor and its Chat Generator."""
        return default_to_dict(
            self,
            chat_generator=component_to_dict(obj=self.chat_generator, name="chat_generator"),
            min_keep_steps=self.min_keep_steps,
            max_summary_tokens=self.max_summary_tokens,
            summary_instruction=self.summary_instruction,
            raise_on_failure=self.raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SummarizationCompactor":
        """Deserialize the compactor and reconstruct its Chat Generator."""
        init_params = data.get("init_parameters", {})
        if init_params.get("chat_generator") is not None:
            deserialize_component_inplace(data=init_params, key="chat_generator")
        return default_from_dict(cls=cls, data=data)
