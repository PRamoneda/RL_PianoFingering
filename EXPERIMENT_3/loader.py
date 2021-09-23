import itertools


KEY_TO_SEMITONE = {'c': 0, 'c#': 1, 'db': 1, 'd': 2, 'd#': 3, 'eb': 3, 'e': 4,
                   'f': 5, 'f#': 6, 'gb': 6, 'g': 7, 'g#': 8, 'ab': 8, 'a': 9,
                   'a#': 10, 'bb': 10, 'b': 11, 'x': None}


def parse_note(note):
    n = KEY_TO_SEMITONE[note[:-1].lower()]
    octave = int(note[-1]) + 1
    return octave * 12 + n - 21


# print(load_all_jp_pieces())
ADAM1 = [55, 58, 53, 53, 51, 53, 55, 51, 55, 58, 53, 53, 51, 53, 51, 53, 53, 55, 55, 56, 55, 53, 58, 53, 53, 55, 55, 56,
         55, 53, 55, 58, 53, 53, 51, 53, 55, 51, 55, 58, 53, 53, 51, 53, 51]


translateADAM1 = {
    51: 0,
    53: 1,
    55: 2,
    56: 3,
    58: 4
}


def load_test3(times=1):
    pieces = []
    for _ in range(times):
        pieces.append([translateADAM1[a] for a in ADAM1])
    return pieces