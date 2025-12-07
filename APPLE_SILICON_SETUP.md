# Apple Silicon Setup Guide for LCZero Training

This guide details how to set up and run the `lczero-training` pipeline on Apple Silicon (M1/M2/M3) Macs, based on a verified configuration.

## 1. Prerequisites
- **Miniforge** (recommended for Apple Silicon) or Anaconda/Miniconda installed.
- **Training Data**: Extracted training data (V5 format or newer) in the `data/train` directory.

## 2. Environment Setup

Create a new conda environment with Python 3.10:

```bash
conda create -n lczero-training python=3.10
conda activate lczero-training
```

Install the required dependencies. Note the specific versions for compatibility:

```bash
# Install core dependencies
pip install tensorflow==2.9.1 numpy protobuf==3.20.3 pyyaml rich tf-models-official

# Install TensorFlow Addons (required for 'mish' activation)
pip install tensorflow-addons==0.23.0
```

> **Note**: `protobuf` is pinned to `3.20.3` to avoid compatibility issues with TensorFlow 2.9.

## 3. Compilation

Compile the protobuf files using the provided script:

```bash
# From the root of the repository
chmod +x init.sh
./init.sh
```

## 4. Configuration Changes

You need to modify `tf/configs/example.yaml` to be compatible with Apple Silicon (using single precision float32).

**File**: `tf/configs/example.yaml`

1.  **Set Precision**: Change `precision` to `single` (float32). `half` (float16) may cause issues on some MPS backends or CPUs.
    ```yaml
    training:
        precision: single    # Changed from 'float'
    ```

2.  **Remove Unsupported Loss**: Remove `future: 0.1` from `loss_weights` if present, as it is not supported in this codebase version.

## 5. Code Patches

The following patches are required to support older training data formats (V5/V6) and fix runtime errors.

### Patch 1: Support Older Data Formats
**File**: `tf/chunkparser.py`
**Location**: `sample_record` method (around line 530)

Add padding to ensure records match the V7 structure size before appending future data:

```python
            # ... existing code ...
            if thresh_p < 1.0 and random.random() > thresh_p:
                continue

            # [PATCH START] Pad record to V7 size if it's shorter (V5 or V6 format)
            if len(record) < v7_struct.size:
                record = record + b"\x00" * (v7_struct.size - len(record))
            # [PATCH END]

            record += b"".join(probs[idx + 1: idx + 1 + n_future_probs])
            # ... existing code ...
```

### Patch 2: Fix Dataset Unpacking
**File**: `tf/tfprocess.py`
**Locations**: `process_inner_loop`, `calculate_test_summaries`, `calculate_validation_summaries` loops.

The dataset contains 9 elements (including `future_boards`), but the loop unpacking expects 8. Update the unpacking to ignore the last element (`_`).

```python
# Change lines like:
# x, y, z, q, m, st_q, opp_idx, next_idx = next(self.train_iter)
# To:
x, y, z, q, m, st_q, opp_idx, next_idx, _ = next(self.train_iter)

# Make this change in:
# 1. process_inner_loop (around line 1500)
# 2. calculate_test_summaries (around line 1826)
# 3. Validation loop (around line 1887)
```

## 6. Running Training

Once set up, run the training script from the `tf` directory (or root, adjusting paths):

```bash
cd tf
python train.py --cfg ./configs/example.yaml --output ./output/mymodel.txt
```

## 7. Troubleshooting

- **`AttributeError: ... has no attribute 'mish'`**: Ensure `tensorflow-addons` is installed.
- **`ValueError: too many values to unpack`**: Apply Patch 2 in `tfprocess.py`.
- **`AssertionError` in shufflebuffer**: Apply Patch 1 in `chunkparser.py`.
- **`ValueError: Unknown precision`**: Ensure `precision: single` is set in `example.yaml`.
