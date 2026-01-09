# Continuation Method (Deprecated)

**Status**: ❌ Failed - Generated invalid syntax
**Experiments**: Exp 1-4
**Replaced by**: Full assignment method (Exp 5+)

---

## Why It Failed

The continuation method attempted to train the model by splitting each musical piece into **3 continuation examples**:

1. **START** (0% completion): `violinoI = \relative do'' {` → predict first 33%
2. **MIDDLE** (50% completion): header + first 50% → predict next 33%
3. **NEAR-END** (75% completion): header + first 75% → predict final 25%

### Critical Flaws

1. **Structural incompleteness**: Only 34% of training examples had closing braces `}`
2. **Fragmentation broke structure learning**: Model never learned proper syntax boundaries
3. **Loss masking confusion**: Input tokens masked with `-100`, only output contributed to loss

### Dataset Statistics

- Total examples: 8,852 (3 per piece)
- Structurally complete: 34%
- Incomplete closures: 66%

### Generated Output (Garbage)

```lilypond
violinoI = \relative do'' {
sib )mid, mib,( reb la' sib( )mid, fad sol( )fad( reb sib4
sol' )red reb sold sold( )mi, dod( )do, red sol
```

**Problems**:
- Invalid syntax: `sib )mid,`
- Random command insertion: `bassVoice \tuplet`
- Broken constructs: `<<vi>>`, unmatched braces
- Completely unusable output

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

## What Replaced It

**Full Assignment Method (Exp 5+)**: See `scripts/prepare_full_assignment_dataset.py`

- **One complete example per piece** (not 3 fragments)
- **100% structurally complete** (every example ends with `}`)
- **No loss masking** - all tokens contribute equally
- **Result**: Valid, musically coherent LilyPond output

### Comparison

| Metric | Continuation (Exp 4) | Full Assignment (Exp 5) |
|--------|---------------------|------------------------|
| Examples per piece | 3 (fragmented) | 1 (complete) |
| Total dataset | 8,852 | 1,503 |
| Structural completeness | 34% | 100% |
| Loss masking | Yes (input ignored) | No (standard training) |
| Output quality | ❌ Garbage | ✅ Valid music |

---

## Research Insight

**Key finding**: For formal languages, **structural completeness matters more than quantity**.

- 8,852 fragmented examples → Invalid syntax
- 1,503 complete examples → Valid generation (83% fewer examples!)

This applies broadly to code generation, mathematics, and structured text where syntax boundaries are critical.

---

## ⚠️ Do Not Use These Files

These files are preserved for historical/thesis documentation only. **Use the full assignment method for all new experiments.**

See:
- `scripts/prepare_full_assignment_dataset.py`
- `src/lilynorm/stages/tokenization/dataset_standard.py`
- `src/lilynorm/stages/training/train_standard.py`
- `src/lilynorm/stages/training/train_weighted.py`
