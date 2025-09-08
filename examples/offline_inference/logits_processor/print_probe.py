# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal script to validate that a logits processor is being called.

Defines a simple per-request logits processor function that only prints when
invoked, and wires it through SamplingParams. Run this file to see prints.
"""

from vllm import LLM, SamplingParams


def print_probe(token_ids, logits):
    """A no-op logits processor that prints when invoked.

    Args:
      token_ids: List of already generated token ids for the request.
      logits: 1-D tensor of next-token logits for this request.
    """
    print(
        f"[print_probe] called: step_len={len(token_ids)}, logits_shape={tuple(logits.shape)}"
    )
    return logits


def main():
    llm = LLM(model="facebook/opt-125m")

    prompt = "Hello, my name is"
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=5,
        logits_processors=[print_probe],
    )

    outputs = llm.generate([prompt], sampling_params)
    for out in outputs:
        print("Generated:", out.outputs[0].text)


if __name__ == "__main__":
    main()
