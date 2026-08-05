"""The diffusion model itself, behind the backend interface.

Everything before this ticket ran on mocks. This is the implementation that loads LLaDA or Dream
and denoises the masked positions — and it is the only module in the project that needs a GPU.

**Selective filling, not generation.** The published examples for both models start from a block
that is entirely mask and denoise it into new text. That is not what is wanted here: the sequence
arrives with a few steps masked and the rest of the plan already written, and the rest must come
back untouched. The loop below therefore never writes to a position that was not masked — the
only positions it can reveal are the ones that started as the mask token, so a healthy step is
outside its reach by construction rather than by instruction. After the loop the frozen positions
are compared against the input and a mismatch is an error, because a guarantee that is never
checked is a hope.

**The loop.** Masked diffusion models fill a fixed-length sequence by revealing tokens over
several passes, most confident first (low-confidence remasking). One pass predicts every masked
position; the most confident share of them is committed; the rest stay masked for the next pass.
Greedy by default, so a repair is reproducible; sampling is available and recorded when used.

**Not measured here.** This module has never been run against a model — the machine it was
written on has no GPU and the weights were never downloaded. What has been checked is that it
imports and fails cleanly without torch or a GPU, and that the pipeline around it works on the
mock backends. The measurements belong to whoever runs it on a GPU.
"""

from typing import Any

from plan_repair.repair.dllm_backend import DLLMError, FillRequest
from plan_repair.repair.tokenization import decode_spans, masked_token_ids

DEFAULT_STEPS = 64
DEFAULT_TEMPERATURE = 0.0


class TorchDLLMBackend:
    """Loads a masked diffusion model and fills the masked positions with it.

    Loading is deferred to the first :meth:`fill`, so constructing a backend costs nothing and a
    machine without a GPU can still import, type-check and test everything around it.
    """

    def __init__(
        self,
        model_id: str,
        mask_token_id: int,
        *,
        name: str = "torch",
        tokenizer: Any = None,
        steps: int = DEFAULT_STEPS,
        temperature: float = DEFAULT_TEMPERATURE,
        quantize_4bit: bool = True,
        device: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.mask_token_id = mask_token_id
        self.steps = steps
        self.temperature = temperature
        self.quantize_4bit = quantize_4bit
        self.device = device
        self.trust_remote_code = trust_remote_code
        self._tokenizer = tokenizer
        self._model: Any = None
        # Observation of the last fill, in tokens — recorded, never acted on.
        self.last_masked_positions: int | None = None
        self.last_mask_tokens_left: int | None = None

    # --- the interface ---------------------------------------------------------------------

    def fill(self, request: FillRequest) -> dict[str, str | None]:
        if not request.mask.masked_step_ids:
            return {}  # nothing to regenerate, so nothing to load
        if request.alignment is None:
            raise DLLMError(
                f"{self.name}: no token alignment — a diffusion backend needs a tokenizer"
            )

        alignment = request.alignment
        prompt = masked_token_ids(alignment, self.mask_token_id)
        filled = self.denoise(prompt)
        if len(filled) != len(prompt):
            raise DLLMError(
                f"{self.name}: denoising returned {len(filled)} tokens for {len(prompt)}"
            )
        _assert_frozen_positions_untouched(prompt, filled, self.mask_token_id, self.name)
        # Counted before decoding, because a tokenizer that drops special tokens would hide an
        # unfinished sequence. This is the reliable answer to "did denoising run out of passes?".
        self.last_masked_positions = sum(1 for token in prompt if token == self.mask_token_id)
        self.last_mask_tokens_left = sum(1 for token in filled if token == self.mask_token_id)
        return dict(decode_spans(alignment, filled, self._require_tokenizer()))

    # --- the model -------------------------------------------------------------------------

    def denoise(self, token_ids: list[int]) -> list[int]:
        """Reveal the masked positions of ``token_ids``, most confident first.

        The write is the point: ``sequence[position] = ...`` only ever runs for a position drawn
        from ``remaining``, and ``remaining`` starts as exactly the masked positions and only
        shrinks. A healthy step is therefore unreachable from inside this loop, in plain Python,
        without depending on what any model does.

        Only :meth:`predict_pass` touches tensors, which is what lets the loop be exercised
        without torch installed.
        """
        sequence = list(token_ids)
        remaining = [
            position for position, token in enumerate(sequence) if token == self.mask_token_id
        ]
        if not remaining:
            return sequence

        per_pass = max(1, -(-len(remaining) // max(self.steps, 1)))
        while remaining:
            predicted, confidence = self.predict_pass(sequence)
            chosen = select_reveals(confidence, remaining, per_pass)
            for position in chosen:
                sequence[position] = predicted[position]
            revealed = set(chosen)
            remaining = [position for position in remaining if position not in revealed]
        return sequence

    def predict_pass(self, sequence: list[int]) -> tuple[list[int], list[float]]:
        """One forward pass: the most likely token at every position, and how sure the model is."""
        torch = _require_torch()
        model = self._require_model()
        device = next(model.parameters()).device

        with torch.no_grad():
            logits = _logits_of(model(torch.tensor([sequence], dtype=torch.long, device=device)))
            if self.temperature > 0:
                noise = torch.rand_like(logits).clamp_min(1e-9)
                scores = logits / self.temperature - torch.log(-torch.log(noise))
            else:
                scores = logits
            probabilities = torch.softmax(logits.float(), dim=-1)
            predicted = scores.argmax(dim=-1)
            confidence = torch.gather(probabilities, -1, predicted.unsqueeze(-1)).squeeze(-1)

        return (
            [int(token) for token in predicted[0].tolist()],
            [float(value) for value in confidence[0].tolist()],
        )

    def _require_model(self) -> Any:
        if self._model is not None:
            return self._model
        torch = _require_torch()
        if self.device is None and not torch.cuda.is_available():
            raise DLLMError(
                f"{self.name}: no CUDA device available. Diffusion inference needs a GPU; "
                f"run this on a GPU machine, or use a mock backend for the pipeline."
            )
        try:
            from transformers import AutoModel
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on the environment
            raise DLLMError(f"{self.name}: transformers is not installed") from exc

        options: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "dtype": torch.bfloat16,
        }
        if self.quantize_4bit:
            options["quantization_config"] = _quantization_config()
            options["device_map"] = "auto"
        elif self.device:
            options["device_map"] = self.device

        try:
            self._model = AutoModel.from_pretrained(self.model_id, **options).eval()
        except Exception as exc:
            raise DLLMError(f"{self.name}: could not load {self.model_id}: {exc}") from exc
        return self._model

    def _require_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on the environment
            raise DLLMError(f"{self.name}: transformers is not installed") from exc
        load: Any = AutoTokenizer.from_pretrained
        self._tokenizer = load(self.model_id, trust_remote_code=self.trust_remote_code)
        return self._tokenizer

    def settings(self) -> dict[str, Any]:
        """What the run was configured with — recorded alongside every result."""
        return {
            "backend": self.name,
            "model_id": self.model_id,
            "mask_token_id": self.mask_token_id,
            "steps": self.steps,
            "temperature": self.temperature,
            "quantize_4bit": self.quantize_4bit,
            "decoding": "greedy" if self.temperature == 0 else "sampled",
        }

    def diagnostics(self) -> dict[str, Any]:
        """Token-level facts about the last fill.

        ``mask_tokens_left`` above zero means denoising ended with the sequence still partly
        masked — the passes ran out before the positions did.
        """
        return {
            "masked_positions": self.last_masked_positions,
            "mask_tokens_left": self.last_mask_tokens_left,
            "steps": self.steps,
        }


def select_reveals(confidence: list[float], remaining: list[int], count: int) -> list[int]:
    """The ``count`` still-masked positions the model is surest about.

    Candidates come from ``remaining`` alone — a position that was never masked cannot be chosen
    however confident the model is about it. Ties break on position, so a run is reproducible.
    """
    ranked = sorted(remaining, key=lambda position: (-confidence[position], position))
    return sorted(ranked[: max(count, 0)])


def _assert_frozen_positions_untouched(
    prompt: list[int], filled: list[int], mask_token_id: int, name: str
) -> None:
    """Every position that was not masked has to come back exactly as it went in."""
    changed = [
        index
        for index, (before, after) in enumerate(zip(prompt, filled, strict=True))
        if before != mask_token_id and before != after
    ]
    if changed:
        raise DLLMError(
            f"{name}: denoising rewrote {len(changed)} position(s) outside the mask "
            f"(first at {changed[0]})"
        )


def _logits_of(output: Any) -> Any:
    logits = getattr(output, "logits", None)
    if logits is None:
        raise DLLMError("the model returned no logits")
    return logits


def _quantization_config() -> Any:
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DLLMError("4-bit loading needs a transformers with BitsAndBytesConfig") from exc
    torch = _require_torch()
    build: Any = BitsAndBytesConfig
    return build(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise DLLMError(
            "torch is not installed. Install the GPU extra to run diffusion inference: "
            "pip install -e '.[gpu]'"
        ) from exc
    return torch


def torch_available() -> bool:
    """Whether inference could run here at all."""
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())


__all__ = [
    "DEFAULT_STEPS",
    "DEFAULT_TEMPERATURE",
    "TorchDLLMBackend",
    "select_reveals",
    "torch_available",
]
