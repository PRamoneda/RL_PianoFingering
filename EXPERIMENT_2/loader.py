

KEY_TO_SEMITONE = {'c': 0, 'c#': 1, 'db': 1, 'd': 2, 'd#': 3, 'eb': 3, 'e': 4,
                   'f': 5, 'f#': 6, 'gb': 6, 'g': 7, 'g#': 8, 'ab': 8, 'a': 9,
                   'a#': 10, 'bb': 10, 'b': 11, 'x': None}


def parse_note(note):
    n = KEY_TO_SEMITONE[note[:-1].lower()]
    octave = int(note[-1]) + 1
    return octave * 12 + n - 21


TEST2 = [0, 1, 2, 3, 4, 3, 2, 1, 0]


def load_test2(times=1):
    pieces = []
    for _ in range(times):
        pieces.append(TEST2)
    return pieces