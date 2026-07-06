from typing import Any, Optional, Union

from transformers.tokenization_utils_base import BatchEncoding

from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer_base import TokenizerBase
from vllm.utils.collection_utils import is_list_of

logger = init_logger(__name__)


class BPEQwenTokenizer(TokenizerBase):
    """Wrapper for bpe-qwen tokenizer to make it compatible with vLLM."""

    def __init__(self, tokenizer: Any) -> None:
        """
        Args:
            tokenizer: The bpe-qwen AutoLinearTokenizer instance
        """
        self.bpe_qwen = tokenizer
        self._vocab = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else {}
        self._vocab_size = len(self._vocab) if self._vocab else getattr(tokenizer, 'vocab_size', 0)
        self._max_token_id = self._vocab_size - 1

    @classmethod
    def from_pretrained(
        cls, path_or_repo_id: str, *, revision: Optional[str] = None
    ) -> "BPEQwenTokenizer":
        """Load tokenizer from pretrained model."""
        from bpe_qwen import AutoLinearTokenizer

        bpe_qwen_tokenizer = AutoLinearTokenizer.from_pretrained(
            path_or_repo_id, revision=revision
        )
        return cls(bpe_qwen_tokenizer)

    # Properties required by vLLM
    @property
    def all_special_tokens_extended(self) -> list[str]:
        if hasattr(self.bpe_qwen, 'all_special_tokens'):
            return self.bpe_qwen.all_special_tokens
        return []

    @property
    def all_special_tokens(self) -> list[str]:
        return self.all_special_tokens_extended

    @property
    def all_special_ids(self) -> list[int]:
        if hasattr(self.bpe_qwen, 'all_special_ids'):
            return self.bpe_qwen.all_special_ids
        return []

    @property
    def bos_token_id(self) -> int:
        return getattr(self.bpe_qwen, 'bos_token_id', 0)

    @property
    def eos_token_id(self) -> int:
        return getattr(self.bpe_qwen, 'eos_token_id', 0)

    @property
    def pad_token_id(self) -> Optional[int]:
        return getattr(self.bpe_qwen, 'pad_token_id', None)

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def max_token_id(self) -> int:
        return self._max_token_id

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: Union[str, list[str], list[int]],
        text_pair: Optional[str] = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        **kwargs  # 接受但忽略其他参数(如 padding)
    ):
        """Tokenize text and return BatchEncoding."""
        input_ids: Union[list[int], list[list[int]]]

        # For list[str], batch of texts
        if is_list_of(text, str):
            input_ids_: list[list[int]] = []
            for p in text:
                each_input_ids = self.encode_one(p, truncation, max_length)
                input_ids_.append(each_input_ids)
            input_ids = input_ids_
        # For list[int], already tokenized
        elif is_list_of(text, int):
            input_ids = text
        # For str, single text
        else:
            input_ids = self.encode_one(text, truncation, max_length)

        # 构建完整的 BatchEncoding
        result = {"input_ids": input_ids}

        # 添加 attention_mask
        if isinstance(input_ids[0], list):
            result["attention_mask"] = [[1] * len(ids) for ids in input_ids]
        else:
            result["attention_mask"] = [1] * len(input_ids)

        return BatchEncoding(result)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def get_added_vocab(self) -> dict[str, int]:
        # bpe-qwen tokenizers have no added vocabulary
        return {}

    def encode_one(
        self,
        text: str,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> list[int]:
        """Encode a single text."""
        input_ids = self.encode(text)

        if truncation and max_length:
            input_ids = input_ids[:max_length]
        return input_ids

    def encode(
        self,
        text: str,
        truncation: Optional[bool] = None,
        max_length: Optional[int] = None,
        add_special_tokens: Optional[bool] = None,
    ) -> list[int]:
        """Encode text to token IDs."""
        return self.bpe_qwen.encode(text)

    def decode(
        self, ids: Union[list[int], int], skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs to text."""
        if isinstance(ids, int):
            ids = [ids]
        return self.bpe_qwen.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        sequences: list[list[int]],
        skip_special_tokens: bool = True
    ) -> list[str]:
        """Batch decode token IDs to texts."""
        if hasattr(self.bpe_qwen, 'batch_decode'):
            return self.bpe_qwen.batch_decode(sequences, skip_special_tokens=skip_special_tokens)
        return [self.decode(seq, skip_special_tokens) for seq in sequences]

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Convert token IDs to tokens."""
        if hasattr(self.bpe_qwen, 'convert_ids_to_tokens'):
            return self.bpe_qwen.convert_ids_to_tokens(ids, skip_special_tokens)
        # Fallback: decode each ID individually
        return [self.bpe_qwen.decode([id]) for id in ids]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Convert tokens to string."""
        return "".join(tokens)

    # Required abstract properties
    @property
    def sep_token(self) -> str:
        raise NotImplementedError()

    @property
    def pad_token(self) -> str:
        raise NotImplementedError()

    def apply_chat_template(
        self,
        messages: list,
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> list[int]:
        # 修正:使用 self.bpe_qwen 而不是 self.tokenizer
        if hasattr(self.bpe_qwen, 'apply_chat_template'):
            return self.bpe_qwen.apply_chat_template(messages, tools=tools, **kwargs)
        else:
            raise NotImplementedError("AutoLinearTokenizer does not support apply_chat_template")
