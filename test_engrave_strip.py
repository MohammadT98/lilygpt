#!/usr/bin/env python3
"""Quick test script for engrave_strip with feature flags."""

from lilynorm.stages.preprocessing import engrave_strip
from lilynorm.utils.options import NormOptions

# Read test file
test_file = r"data\normalized_dataset\NO PUB codifica_461\NO PUB codifica\lotti_concerto_oboe_d'amore_score_part1.ly"

print(f"Testing engrave_strip on: {test_file}")
print("=" * 80)

with open(test_file, 'r', encoding='utf-8') as f:
    original = f.read()

# Create options with keep_engraving=True (safe mode)
opts = NormOptions()
opts.keep_engraving = True

# Run engrave_strip
result = engrave_strip.run(original, opts)

# Show statistics
print(f"\nOriginal size: {len(original)} chars, {len(original.splitlines())} lines")
print(f"Result size:   {len(result)} chars, {len(result.splitlines())} lines")
print(f"Reduction:     {len(original) - len(result)} chars ({100 * (1 - len(result)/len(original)):.1f}%)")

# Show first 50 lines of output
print("\n" + "=" * 80)
print("First 50 lines of output:")
print("=" * 80)
for i, line in enumerate(result.split('\n')[:50], 1):
    print(f"{i:3d}: {line}")

# Save to file for inspection
output_file = "test_engrave_strip_output.ly"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(result)

print("\n" + "=" * 80)
print(f"Full output saved to: {output_file}")
print("You can compile it with: lilypond test_engrave_strip_output.ly")
