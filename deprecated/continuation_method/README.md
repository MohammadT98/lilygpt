# Continuation Method (Deprecated)

**Experiments**: 1-4
**Status**: Failed - Invalid syntax generation
**Superseded by**: Full assignment method (Experiments 5+)

---

## Failure Analysis

The continuation method attempted to train the model by splitting each musical piece into **3 continuation examples**:

1. **START** (0% completion): `violinoI = \relative do'' {` → predict first 33%
2. **MIDDLE** (50% completion): header + first 50% → predict next 33%
3. **NEAR-END** (75% completion): header + first 75% → predict final 25%

### Critical Issues

1. **Structural incompleteness**: Only 34% of training examples contained closing braces
2. **Fragment-based learning**: Three-way splitting prevented the model from learning complete syntactic structures
3. **Masked loss function**: Input masking (`-100` labels) created ambiguity in boundary token prediction

### Dataset Statistics

- Total examples: 48,502 (actual generated set; far more than the initial 8,852 plan)
- Structurally complete: 34%
- Incomplete closures: 66%

### Representative Output

```lilypond
violinoI = \relative do'' {
sib )mid, mib,( reb la' sib( )mid, fad sol( )fad( reb sib4
sol' )red reb sold sold( )mi, dod( )do, red sol
```

**Observed defects**:
- Malformed tokens: `sib )mid,`
- Spurious command insertion: `bassVoice \tuplet`
- Unbalanced delimiters: `<<vi>>`, orphaned braces
- Non-compilable output

---

## Files in This Directory

### Scripts (`scripts/`)
- `prepare_continuation_dataset.py` - Generates continuation dataset with 3 splits per piece
- `prepare_continuation_with_log.py` - Wrapper with logging

### Source Code (`src/`)
- `dataset_continuation.py` - `LilyContinuationDataset` class with masked loss
- `train.py` - Continuation training script (masked input tokens)

### SLURM Jobs (`slurm/`)
- `train_exp4_standard.slurm` - Experiment 4 job script

### Batch Scripts (`bin/`)
- `prepare_continuation_data.bat` - Windows batch wrapper

---

## Replacement Approach

**Full Assignment Method (Experiments 5+)**: See `scripts/prepare_full_assignment_dataset.py`

- One complete example per musical assignment (no fragmentation)
- 100% structural completeness (all examples properly closed)
- Standard loss computation (no input masking)
- Result: Syntactically valid, musically coherent output

### Comparison

| Metric | Continuation (Exp 4) | Full Assignment (Exp 5) |
|--------|---------------------|------------------------|
| Examples per piece | 3 planned (fragmented) | 1 (complete) |
| Total dataset | 48,502 actual (inflated) | 1,503 |
| Structural completeness | 34% | 100% |
| Loss masking | Yes (input ignored) | No (standard training) |
| Output quality | Invalid | Valid |

---

## Key Finding

For formal language generation, **structural completeness is more critical than dataset size**.

- 8,852 fragmented examples → Invalid syntax generation
- 1,503 complete examples → Valid output (83% dataset reduction)

This principle generalizes to other formal languages (code, mathematics, structured markup) where syntactic boundaries are semantically significant.

---

## Usage Note

These files are maintained for research documentation purposes only. Current work should use the full assignment method.

See:
- `scripts/prepare_full_assignment_dataset.py`
- `src/lilynorm/stages/tokenization/dataset_standard.py`
- `src/lilynorm/stages/training/train_standard.py`
- `src/lilynorm/stages/training/train_weighted.py`
