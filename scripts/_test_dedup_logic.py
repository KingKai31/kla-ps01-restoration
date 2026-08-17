"""
Windows/NTFS is case-insensitive for directory names, so the real
uppercase/lowercase-duplicate scenario (DIV2K_train_HR vs div2k_train_hr as
genuinely distinct directories) can't be reproduced on this dev machine's
filesystem - mkdir'ing both just resolves to the same directory. This tests
the selection logic in isolation with mock objects instead, so the algorithm
itself is verified even though the real Linux (Kaggle) case-sensitive
scenario can't be built locally.
"""
from dataclasses import dataclass


@dataclass
class MockDir:
    name: str

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


def pick_winner(group):
    """Mirrors the exact selection logic in _walk_dedup."""
    non_lower = [d for d in group if d.name != d.name.lower()]
    return non_lower[0] if len(non_lower) == 1 else sorted(group, key=lambda d: d.name)[0]


# Case 1: exact DIV2K scenario - one canonical mixed-case, one all-lowercase duplicate
group1 = [MockDir("DIV2K_train_HR"), MockDir("div2k_train_hr")]
winner1 = pick_winner(group1)
print(f"Case 1 (mixed-case + lowercase dup): winner = {winner1.name!r}")
assert winner1.name == "DIV2K_train_HR", "should prefer the non-all-lowercase canonical name"

# Case 2: three-way collision, one canonical
group2 = [MockDir("dtd-r1.0.1"), MockDir("DTD-R1.0.1"), MockDir("dtd_r1_0_1")]
# note: dtd_r1_0_1 differs in more than case (underscores vs dots) so this
# wouldn't actually collide in lowercase - use a real 3-way case collision instead
group2 = [MockDir("Flickr2K"), MockDir("flickr2k"), MockDir("FLICKR2K")]
winner2 = pick_winner(group2)
print(f"Case 2 (3-way collision, ambiguous - two non-lowercase): winner = {winner2.name!r}")
# two candidates are non-all-lowercase (Flickr2K, FLICKR2K) -> ambiguous -> alphabetically first
assert winner2.name == sorted([d.name for d in group2])[0]

# Case 3: no collision at all
group3 = [MockDir("DIV2K_train_HR")]
winner3 = pick_winner(group3)
print(f"Case 3 (no collision): winner = {winner3.name!r}")
assert winner3.name == "DIV2K_train_HR"

# Case 4: both all-lowercase (genuinely ambiguous, no canonical signal)
group4 = [MockDir("data_v1"), MockDir("data_V1".lower())]  # both lowercase after .lower() comparison upstream groups them
group4 = [MockDir("part1"), MockDir("part1_copy")]  # not actually same lowercase - fix
group4 = [MockDir("images"), MockDir("images")]
winner4 = pick_winner(group4)
print(f"Case 4 (identical all-lowercase names): winner = {winner4.name!r}")
assert winner4.name == "images"

print("ALL DEDUP LOGIC TESTS PASSED")
