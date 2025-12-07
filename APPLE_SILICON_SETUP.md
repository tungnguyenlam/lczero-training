# Apple Silicon Setup Guide for LCZero Training

This guide details how to set up and run the `lczero-training` pipeline on Apple Silicon (M1/M2/M3/M4) Macs with **MPS GPU acceleration**.

## 1. Prerequisites
- **Miniforge** (recommended for Apple Silicon) or Anaconda/Miniconda installed.
- **Training Data**: Extracted training data (V5 format or newer) in the `data/train` directory.

## 2. Environment Setup

Create a new conda environment with Python 3.10:

```bash
conda create -n lczero-training python=3.10
conda activate lczero-training
```

Install the required dependencies with MPS support:

```bash
# Install Apple Silicon optimized TensorFlow with Metal GPU support
pip install tensorflow-macos==2.9.2 tensorflow-metal==0.5.1

# Install other dependencies
pip install numpy protobuf==3.20.3 pyyaml rich tf-models-official

# Install TensorFlow Addons (required for 'mish' activation)
pip install tensorflow-addons==0.19.0
```

> **Note**: `protobuf` is pinned to `3.20.3` to match the system protoc version.

## 3. Compilation

Compile the protobuf files using the provided script:

```bash
# From the root of the repository
chmod +x init.sh
./init.sh
```

## 4. Configuration Changes

Modify `tf/configs/example.yaml`:

1. **Set Precision**: Change `precision` to `single` (float32).
    ```yaml
    training:
        precision: single
    ```

2. **Remove Unsupported Loss**: Remove `future: 0.1` from `loss_weights` if present.

## 5. Code Patches

### Patch 1: Support Older Data Formats
**File**: `tf/chunkparser.py` (around line 530)

```python
            # Pad record to V7 size if it's shorter (V5 or V6 format)
            if len(record) < v7_struct.size:
                record = record + b"\x00" * (v7_struct.size - len(record))
```

### Patch 2: Fix Dataset Unpacking
**File**: `tf/tfprocess.py` - Update unpacking to ignore the 9th element:

```python
x, y, z, q, m, st_q, opp_idx, next_idx, _ = next(self.train_iter)
```

Apply in: `train_step` (~line 1501), `calculate_test_summaries` (~line 1826), validation loop (~line 1887).

## 6. Running Training

```bash
python ./tf/train.py --cfg ./tf/configs/example.yaml --output ./output/mymodel.txt
```

You should see:
```
Metal device set to: Apple M4
systemMemory: 16.00 GB
maxCacheSize: 5.92 GB
```

## 7. Troubleshooting

- **`ImportError: cannot import name 'builder'`**: Run `pip install protobuf==3.20.3` and `./init.sh`
- **`AttributeError: ... 'mish'`**: Install `tensorflow-addons`
- **`ValueError: too many values to unpack`**: Apply Patch 2
- **`AssertionError` in shufflebuffer**: Apply Patch 1
