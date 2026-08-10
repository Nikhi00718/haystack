# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable
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
    _messages_except,
)
from haystack.token_counters import TokenCounter
from haystack.token_counters.utils import _render_message
from haystack.utils.async_utils import _execute_component_async
from haystack.utils.deserialization import deserialize_component_inplace
from haystack.utils.experimental import _experimental

logger = logging.getLogger(__name__)

_STRATEGY = "summarization"

_DEFAULT_SUMMARY_INSTRUCTION = """You are compacting part of a conversation between a user and an AI agent so the \
agent can keep working with fewer tokens. Write a self-contained summary that preserves:
- The user's goal, requirements, constraints, and preferences.
- Decisions and the reasoning behind them.
- Work already completed and important tool results.
- Exact file paths, URLs, identifiers, and references to stored data.
- Unresolved work and the immediate next step.

Fold any existing <conversation_summary> blocks into one summary. Record only what the conversation shows. Do not \
infer or add advice. Use plain prose or short bullets, and do not address the user."""


def _indices(messages: list[ChatMessage], start: int, end: int, *, summaries: bool) -> list[int]:
    """Return summary or non-summary indices in a bounded part of a conversation."""
    return [
        index
        for index in range(start, end)
        if _is_compaction_message(message=messages[index], strategy=_STRATEGY) is summaries
    ]


def _raw_historical_turn_groups(
    messages: list[ChatMessage], system_end: int, task_index: int | None
) -> list[list[int]]:
    """Return shared historical-turn groups with this strategy's summaries filtered out."""
    return [
        [index for index in group if not _is_compaction_message(message=messages[index], strategy=_STRATEGY)]
        for group in _historical_turn_groups(messages=messages, system_end=system_end, task_index=task_index)
    ]


def _groups_to_summarize(
    messages: list[ChatMessage],
    groups: list[list[int]],
    target_tokens: int,
    summary_budget: int,
    token_counter: TokenCounter,
) -> list[int]:
    """Select the fewest oldest groups that should make room for a summary of the configured size."""
    selected: list[int] = []
    for group in groups:
        selected.extend(group)
        if (
            token_counter.count(messages=_messages_except(messages=messages, indices=selected)) + summary_budget
            <= target_tokens
        ):
            break
    return selected


def _summary_message(text: str, summarized_messages: int, source: str) -> ChatMessage:
    """Build a marked summary message."""
    body = f"<conversation_summary>\n{text.strip()}\n</conversation_summary>"
    meta = {_COMPACTION_META_KEY: {"strategy": _STRATEGY, "summarized_messages": summarized_messages, "source": source}}
    return ChatMessage.from_user(text=body, meta=meta)


def _replace_indices(messages: list[ChatMessage], indices: list[int], summary: ChatMessage) -> list[ChatMessage]:
    """Replace possibly non-contiguous selected messages with one summary at their first position."""
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
        if token_counter.count(messages=messages) <= target_tokens:
            return None
        budget, generation_kwargs = _resolve_output_token_limit(
            chat_generator=self.chat_generator, default_limit=self.max_summary_tokens
        )

        def generate(prompt: list[ChatMessage]) -> dict[str, Any]:
            kwargs: dict[str, Any] = {"messages": prompt}
            if generation_kwargs is not None:
                kwargs["generation_kwargs"] = generation_kwargs
            return self.chat_generator.run(**kwargs)

        return self._compact(
            original=messages,
            target_tokens=target_tokens,
            token_counter=token_counter,
            budget=budget,
            generate=generate,
        )

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
        if token_counter.count(messages=messages) <= target_tokens:
            return None
        budget, generation_kwargs = _resolve_output_token_limit(
            chat_generator=self.chat_generator, default_limit=self.max_summary_tokens
        )

        async def generate(prompt: list[ChatMessage]) -> dict[str, Any]:
            kwargs: dict[str, Any] = {"messages": prompt}
            if generation_kwargs is not None:
                kwargs["generation_kwargs"] = generation_kwargs
            return await _execute_component_async(component_instance=self.chat_generator, **kwargs)

        return await self._compact_async(
            original=messages,
            target_tokens=target_tokens,
            token_counter=token_counter,
            budget=budget,
            generate=generate,
        )

    def _prompt(self, messages: list[ChatMessage], budget: int) -> list[ChatMessage]:
        """Build the bounded summarization instruction and rendered source transcript."""
        transcript = "\n".join(_render_message(message=message) for message in messages)
        instruction = (
            f"{self.summary_instruction}\n\nWrite a complete summary in no more than approximately {budget} tokens. "
            "Prioritize completeness within that limit so the response is not cut off."
        )
        return [
            ChatMessage.from_system(text=instruction),
            ChatMessage.from_user(text=f"<conversation_to_summarize>\n{transcript}\n</conversation_to_summarize>"),
        ]

    def _apply_result(
        self,
        messages: list[ChatMessage],
        indices: list[int],
        source: str,
        result: dict[str, Any],
        token_counter: TokenCounter,
    ) -> list[ChatMessage]:
        """Validate a generator reply and replace its source messages when the result is smaller."""
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

    def _attempt(
        self,
        messages: list[ChatMessage],
        indices: list[int],
        source: str,
        budget: int,
        token_counter: TokenCounter,
        generate: Callable[[list[ChatMessage]], dict[str, Any]],
    ) -> list[ChatMessage] | None:
        """Attempt one synchronous summary, applying the configured failure policy."""
        try:
            result = generate(self._prompt(messages=[messages[index] for index in indices], budget=budget))
            return self._apply_result(
                messages=messages, indices=indices, source=source, result=result, token_counter=token_counter
            )
        except Exception as error:
            if self.raise_on_failure:
                raise
            logger.warning(
                "Summarizing the conversation for context compaction failed; keeping the last successful result. "
                "Error: {error}",
                error=error,
            )
            return None

    async def _attempt_async(
        self,
        messages: list[ChatMessage],
        indices: list[int],
        source: str,
        budget: int,
        token_counter: TokenCounter,
        generate: Callable[[list[ChatMessage]], Awaitable[dict[str, Any]]],
    ) -> list[ChatMessage] | None:
        """Attempt one asynchronous summary, applying the configured failure policy."""
        try:
            result = await generate(self._prompt(messages=[messages[index] for index in indices], budget=budget))
            return self._apply_result(
                messages=messages, indices=indices, source=source, result=result, token_counter=token_counter
            )
        except Exception as error:
            if self.raise_on_failure:
                raise
            logger.warning(
                "Summarizing the conversation for context compaction failed; keeping the last successful result. "
                "Error: {error}",
                error=error,
            )
            return None

    def _compact(
        self,
        original: list[ChatMessage],
        target_tokens: int,
        token_counter: TokenCounter,
        budget: int,
        generate: Callable[[list[ChatMessage]], dict[str, Any]],
    ) -> list[ChatMessage] | None:
        """Run synchronous historical, consolidation, and current-step compaction tiers in order."""
        working = list(original)

        # First replace the fewest oldest raw historical turns expected to reach the target.
        while token_counter.count(messages=working) > target_tokens:
            system_end = _leading_system_end(messages=working)
            task_index = _latest_user_index(messages=working)
            groups = [
                group
                for group in _raw_historical_turn_groups(messages=working, system_end=system_end, task_index=task_index)
                if group
            ]
            if not groups:
                break
            selected = _groups_to_summarize(
                messages=working,
                groups=groups,
                target_tokens=target_tokens,
                summary_budget=budget,
                token_counter=token_counter,
            )
            compacted = self._attempt(
                messages=working,
                indices=selected,
                source="historical_turns",
                budget=budget,
                token_counter=token_counter,
                generate=generate,
            )
            if compacted is None:
                return self._result(original=original, working=working, token_counter=token_counter)
            working = compacted

        # If all historical turns are summaries and the target is still unmet, fold them into one summary.
        if token_counter.count(messages=working) > target_tokens:
            summaries = self._summary_indices(messages=working, source="historical_summaries")
            if len(summaries) > 1:
                compacted = self._attempt(
                    messages=working,
                    indices=summaries,
                    source="historical_summaries",
                    budget=budget,
                    token_counter=token_counter,
                    generate=generate,
                )
                if compacted is None:
                    return self._result(original=original, working=working, token_counter=token_counter)
                working = compacted

        # Finally summarize the oldest eligible agent steps while preserving the configured recent steps.
        while token_counter.count(messages=working) > target_tokens:
            system_end = _leading_system_end(messages=working)
            task_index = _latest_user_index(messages=working)
            step_groups = _current_step_groups(messages=working, system_end=system_end, task_index=task_index)
            eligible = step_groups[: max(len(step_groups) - self.min_keep_steps, 0)]
            if not eligible:
                break

            # Fold accumulated current-task summaries before consuming more raw agent steps.
            summaries = self._summary_indices(messages=working, source="current_task_summaries")
            if len(summaries) > 1:
                compacted = self._attempt(
                    messages=working,
                    indices=summaries,
                    source="current_task_summaries",
                    budget=budget,
                    token_counter=token_counter,
                    generate=generate,
                )
                if compacted is None:
                    return self._result(original=original, working=working, token_counter=token_counter)
                working = compacted

            # Recompute positions after consolidation, then select the minimum useful prefix of raw steps.
            system_end = _leading_system_end(messages=working)
            task_index = _latest_user_index(messages=working)
            step_groups = _current_step_groups(messages=working, system_end=system_end, task_index=task_index)
            eligible = step_groups[: max(len(step_groups) - self.min_keep_steps, 0)]
            selected = _groups_to_summarize(
                messages=working,
                groups=eligible,
                target_tokens=target_tokens,
                summary_budget=budget,
                token_counter=token_counter,
            )
            compacted = self._attempt(
                messages=working,
                indices=selected,
                source="current_task_steps",
                budget=budget,
                token_counter=token_counter,
                generate=generate,
            )
            if compacted is None:
                break
            working = compacted
        return self._result(original=original, working=working, token_counter=token_counter)

    async def _compact_async(
        self,
        original: list[ChatMessage],
        target_tokens: int,
        token_counter: TokenCounter,
        budget: int,
        generate: Callable[[list[ChatMessage]], Awaitable[dict[str, Any]]],
    ) -> list[ChatMessage] | None:
        """Run asynchronous historical, consolidation, and current-step compaction tiers in order."""
        working = list(original)

        # First replace the fewest oldest raw historical turns expected to reach the target.
        while token_counter.count(messages=working) > target_tokens:
            system_end = _leading_system_end(messages=working)
            task_index = _latest_user_index(messages=working)
            groups = [
                group
                for group in _raw_historical_turn_groups(messages=working, system_end=system_end, task_index=task_index)
                if group
            ]
            if not groups:
                break
            selected = _groups_to_summarize(
                messages=working,
                groups=groups,
                target_tokens=target_tokens,
                summary_budget=budget,
                token_counter=token_counter,
            )
            compacted = await self._attempt_async(
                messages=working,
                indices=selected,
                source="historical_turns",
                budget=budget,
                token_counter=token_counter,
                generate=generate,
            )
            if compacted is None:
                return self._result(original=original, working=working, token_counter=token_counter)
            working = compacted

        # If all historical turns are summaries and the target is still unmet, fold them into one summary.
        if token_counter.count(messages=working) > target_tokens:
            summaries = self._summary_indices(messages=working, source="historical_summaries")
            if len(summaries) > 1:
                compacted = await self._attempt_async(
                    messages=working,
                    indices=summaries,
                    source="historical_summaries",
                    budget=budget,
                    token_counter=token_counter,
                    generate=generate,
                )
                if compacted is None:
                    return self._result(original=original, working=working, token_counter=token_counter)
                working = compacted

        # Finally summarize the oldest eligible agent steps while preserving the configured recent steps.
        while token_counter.count(messages=working) > target_tokens:
            system_end = _leading_system_end(messages=working)
            task_index = _latest_user_index(messages=working)
            step_groups = _current_step_groups(messages=working, system_end=system_end, task_index=task_index)
            eligible = step_groups[: max(len(step_groups) - self.min_keep_steps, 0)]
            if not eligible:
                break

            # Fold accumulated current-task summaries before consuming more raw agent steps.
            summaries = self._summary_indices(messages=working, source="current_task_summaries")
            if len(summaries) > 1:
                compacted = await self._attempt_async(
                    messages=working,
                    indices=summaries,
                    source="current_task_summaries",
                    budget=budget,
                    token_counter=token_counter,
                    generate=generate,
                )
                if compacted is None:
                    return self._result(original=original, working=working, token_counter=token_counter)
                working = compacted

            # Recompute positions after consolidation, then select the minimum useful prefix of raw steps.
            system_end = _leading_system_end(messages=working)
            task_index = _latest_user_index(messages=working)
            step_groups = _current_step_groups(messages=working, system_end=system_end, task_index=task_index)
            eligible = step_groups[: max(len(step_groups) - self.min_keep_steps, 0)]
            selected = _groups_to_summarize(
                messages=working,
                groups=eligible,
                target_tokens=target_tokens,
                summary_budget=budget,
                token_counter=token_counter,
            )
            compacted = await self._attempt_async(
                messages=working,
                indices=selected,
                source="current_task_steps",
                budget=budget,
                token_counter=token_counter,
                generate=generate,
            )
            if compacted is None:
                break
            working = compacted
        return self._result(original=original, working=working, token_counter=token_counter)

    def _summary_indices(self, messages: list[ChatMessage], source: str) -> list[int]:
        """Return historical or current-task summary indices based on their conversation position."""
        system_end = _leading_system_end(messages=messages)
        task_index = _latest_user_index(messages=messages)
        end = task_index if task_index is not None else len(messages)
        if source == "historical_summaries":
            return _indices(messages=messages, start=system_end, end=end, summaries=True)
        start = task_index + 1 if task_index is not None else system_end
        return _indices(messages=messages, start=start, end=len(messages), summaries=True)

    @staticmethod
    def _result(
        original: list[ChatMessage], working: list[ChatMessage], token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        """Return partial or complete progress only when it reduced the original conversation."""
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
