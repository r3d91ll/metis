# GPU Configuration Guide

## Quick Reference

### Single GPU
```yaml
embeddings:
  device: "cuda:0"  # Use first GPU
  batch_size: 32    # Adjust based on VRAM (16GB = 32, 24GB = 48, 48GB = 64)
```

### Multi-GPU (Data Parallel)
```yaml
embeddings:
  device: "cuda"    # Use all available GPUs
  batch_size: 64    # Can increase with multiple GPUs
```

### CPU-Only (Fallback)
```yaml
embeddings:
  device: "cpu"
  batch_size: 8     # Much smaller for CPU
```

## Batch Size Recommendations

### GPU Memory → Batch Size Mapping

| VRAM  | Jina v4 Batch Size | Papers/Hour (Est) |
|-------|-------------------|-------------------|
| 16 GB | 16-24             | ~300              |
| 24 GB | 32-48             | ~500              |
| 40 GB | 48-64             | ~800              |
| 80 GB | 64-96             | ~1200             |

**Formula**: `batch_size ≈ (VRAM_GB - 4) * 1.5`

The -4 GB accounts for model weights, the 1.5 multiplier is empirical for Jina v4.

## Multi-GPU Setup

### Automatic GPU Detection
```python
import torch
num_gpus = torch.cuda.device_count()
print(f"Available GPUs: {num_gpus}")
```

### Manual GPU Selection
```yaml
embeddings:
  device: "cuda:0,1,2,3"  # Specify exact GPUs (NOT IMPLEMENTED YET)
```

**Current Limitation**: The import pipeline uses Metis embedder, which currently only supports single device string (`cuda`, `cuda:0`, or `cpu`). For multi-GPU, set `device: "cuda"` and PyTorch will use DataParallel automatically.

## Performance Tuning

### CPU Batch vs GPU Batch

The pipeline has TWO separate batch sizes:

1. **CPU Batch** (`import.batch_size`)
   - Papers processed together for metadata extraction
   - Default: 100
   - Affects: Memory usage, database insert batch size
   - Tune: Higher = more memory but fewer DB round-trips

2. **GPU Batch** (`embeddings.batch_size`)
   - Papers embedded together in single GPU forward pass
   - Default: 32
   - Affects: GPU utilization, VRAM usage
   - Tune: Increase until GPU memory saturates

**Optimal Relationship**: `CPU_batch_size` should be a multiple of `GPU_batch_size` to avoid partial batches.

Example:
```yaml
import:
  batch_size: 96           # CPU: Process 96 papers
  embedding_batch_size: 32 # GPU: Embed in 3 batches of 32

embeddings:
  batch_size: 32           # Must match embedding_batch_size
```

### Streaming vs Batching

The pipeline streams papers from disk (NDJSON format) to avoid loading the entire 4.6GB file into memory:

```plaintext
Disk → Stream (1 paper) → Buffer (96 papers) → Embed (32 at a time) → Insert (96 bulk)
```

## Monitoring GPU Usage

### During Import
```bash
# Terminal 1: Run import
poetry run python experiments/arxiv_import/import_pipeline.py

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi
```

### Key Metrics
- **GPU Utilization**: Should be 80-100% during embedding batches
- **Memory Usage**: Should be near capacity but not OOM
- **Temperature**: Keep below 85°C for sustained performance

### Optimization Signals

| Symptom                  | Problem          | Solution              |
|--------------------------|------------------|-----------------------|
| GPU util < 50%           | Batch too small  | Increase batch_size   |
| OOM errors               | Batch too large  | Decrease batch_size   |
| CPU waiting for GPU      | GPU bottleneck   | Add more GPUs         |
| GPU waiting for CPU      | CPU bottleneck   | Increase CPU batch    |

## Estimated Runtime

### Single A100 (80GB)
- Batch size: 96
- ~1000 papers/hour
- **Total time**: 2400 hours (~100 days) ❌ TOO SLOW

### 4x A100 (80GB each)
- Batch size: 96 per GPU
- ~4000 papers/hour
- **Total time**: 600 hours (~25 days) ❌ STILL TOO SLOW

### Recommended: 16x A100
- Batch size: 96 per GPU
- ~16,000 papers/hour
- **Total time**: 150 hours (~6 days) ✅ ACCEPTABLE

**Reality Check**: With single GPU, expect **15-24 hours** for the full dataset per the user's estimate. This assumes optimizations like:
- Mixed precision (FP16/BF16)
- Flash Attention
- Optimized CUDA kernels
- Minimal CPU overhead

## Testing Before Full Run

Always test with a sample first:

```bash
# Test 1000 papers to validate config
poetry run python experiments/arxiv_import/import_pipeline.py --limit 1000

# Monitor GPU usage and adjust batch_size based on VRAM utilization
# Repeat until GPU memory is 85-95% utilized
```

## Current Configuration

Check `config/arxiv_import.yaml`:
```bash
cat config/arxiv_import.yaml | grep -A 5 "embeddings:"
```
