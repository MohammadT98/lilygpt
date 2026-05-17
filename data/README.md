# `data/`

The `data/` tree is git-ignored. LilyBench's reproduction pipeline expects the
following layout, populated from the Zenodo release that accompanies the paper:

```
data/
  bmdataset/                       # in-domain corpus
    preprocessed/*.ly              # nederlands-pitched, include-resolved scores
    metadata.json                  # per-work composer / period / form / ensemble
  mutopia/                         # out-of-domain corpus
    dataset_mutopia.json           # manifest + relative .ly paths
    stripped/.../...               # .ly files referenced by the manifest
  emopia/                          # emotion-recognition corpus
    manifest.csv                   # one row per clip (clip_id, song_id, label, ly_path)
    ly/*.ly                        # midi2ly-converted LilyPond clips
  splits/                          # deterministic work-level splits over bmdataset
    train.jsonl
    val.jsonl
    test.jsonl                     # the in-domain reference used by FMD & JS
  prompt_bank.jsonl                # output of `lilybench prompt-bank build`
  runs/                            # per-model generation + understanding outputs
```

All of these — the BMdataset corpus, the Mutopia stripped tree, the EMOPIA
manifest, and the BMdataset splits — are published as a single archive on
Zenodo. Unpack it under `data/` and you are ready to reproduce the paper.

## Data preparation scripts (only needed to *rebuild* the archive)

* `scripts/convert_mutopia.py` upgrades a raw Mutopia tree with `convert-ly`
  (the corpus carries ~15 years of LilyPond syntax drift).
* `scripts/prepare_emopia.py` downloads EMOPIA, runs `midi2ly` on every clip,
  and emits the manifest CSV.

End users do not need to run them; they only matter if you want to regenerate
the published artifacts from scratch.
