# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ResponsesRequest
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3 model.

    The Qwen3 model uses <think>...</think> tokens to denote reasoning text
    within its output. The model provides a strict switch to disable reasoning
    output via the 'enable_thinking=False' parameter. This parser extracts the
    reasoning content enclosed by <think> and </think> tokens from the model's
    output.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.
 
        Handles both cases:
        - <think>abc</think>xyz (start token in output)
        - abc</think>xyz (start token was in prompt, not in generated output)
        """
 
        if self.end_token not in model_output:
            if self.start_token in model_output:
                _, _, after = model_output.partition(self.start_token)
                return after or None, None
            return None, model_output
 
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )
 
        reasoning, _, content = model_output.partition(self.end_token)
        final_content = content.strip() if content and content.strip() else None
        return reasoning, final_content
