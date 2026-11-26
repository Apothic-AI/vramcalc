# vramcalc

`vramcalc` estimates how much GPU memory a model needs for inference, and which GPUs can hold it.

It is a Go command-line port of the [GPU VRAM Calculator](https://huggingface.co/spaces/aquanode/gpu-vram-calculator) Space by Aquanode. It uses the Space's formulas, its catalog of 1,333 models, and its list of 29 GPUs.

## Install

Requires Go 1.22 or newer. The tool uses only the standard library.

```sh
go install github.com/Apothic-AI/vramcalc/cmd/vramcalc@latest   # into $(go env GOPATH)/bin
```

To build from a clone instead:

```sh
git clone https://github.com/Apothic-AI/vramcalc && cd vramcalc
go build -o bin/vramcalc ./cmd/vramcalc                          # writes bin/vramcalc
```

## Usage

```sh
vramcalc Qwen/Qwen3-8B                        # a model from the catalog
vramcalc --params 70 --dtype FP16             # any size, in billions of parameters
vramcalc deepseek-ai/DeepSeek-V3 --precision int4
vramcalc models llama-3.1                     # find model IDs
vramcalc models --json qwen3                  # the same search as JSON
vramcalc gpus                                 # list the GPUs
vramcalc --json Qwen/Qwen3-8B                 # estimate as JSON, for scripts and agents
```

Model IDs match exactly, ignoring case. Put the subcommand first: `vramcalc models --json QUERY`.

| Flag | Meaning |
| --- | --- |
| `--params N` | Parameter count in billions, such as `70`, `8.2` or `8.2B`. Use it instead of a model ID. |
| `--dtype D` | Published precision for `--params`: `BF16`, `FP16`, `FP32` or `FP8`. Default `BF16`. |
| `--precision P` | Precision for the GPU table: `native`, `fp8` or `int4`. Default `native`. |
| `--json` | Print JSON instead of tables. |

Exit codes: `0` success, `1` lookup failed (unknown model or no matches), `2` bad arguments.

## How the numbers are calculated

- Weights = parameters × bytes per parameter: FP32 4, FP16 and BF16 2, FP8 1, INT4 0.5.
- Needed to run = weights × 1.2. The factor is a rule of thumb for KV cache, activations and allocator overhead at single-request inference around an 8K context.
- Figures are in GiB (1024³ bytes). The Space labels them GB, but its README says they are GiB, and the CLI uses GiB. GPU VRAM values come from the Space and are compared with these figures directly.
- GPUs needed = ceil(needed ÷ VRAM), at least 1. Counts above 8 are labelled as more than one 8-GPU node. Multi-GPU counts assume an even split across cards.

## Limits

- The 1.2 factor covers single-request inference at about 8K context. Larger batches or longer contexts can need much more, because the KV cache can outgrow the weights.
- Parameter counts are the catalog totals. For mixture-of-experts models, that includes every expert.
- These are estimates. Check a real deployment before you buy hardware.

## Data and attribution

`internal/catalog/catalog.json` is a snapshot of the Space's data from commit `1fd43a0`. The Space's README declares the Apache-2.0 license. The formulas follow the Space's page script. To refresh the catalog, re-extract the `<script id="data">` block from the Space's `index.html`.

## Development

```sh
go vet ./...
go test ./...
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
