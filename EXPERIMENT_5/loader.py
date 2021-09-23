import music21


KEY_TO_SEMITONE = {'c': 0, 'c#': 1, 'db': 1, 'd': 2, 'd#': 3, 'eb': 3, 'e': 4,
                   'f': 5, 'f#': 6, 'gb': 6, 'g': 7, 'g#': 8, 'ab': 8, 'a': 9,
                   'a#': 10, 'bb': 10, 'b': 11, 'x': None}


def parse_note(note):
    n = KEY_TO_SEMITONE[note[:-1].lower()]
    octave = int(note[-1]) + 1
    return octave * 12 + n - 21


translate5 = {
    46: 0,
    48: 1,
    50: 2,
    51: 3,
    53: 4,
    55: 5,
    56: 6,
    58: 7,
}


def load_test5(times=1):
    sc = music21.converter.parse('test5.musicxml')
    rh = [translate5[parse_note(str(n.pitch).lower())] for n in sc.parts[0].flat.getElementsByClass('Note')]
    pieces = []
    for _ in range(times):
        pieces.append(rh)
    return pieces


# print(load_test5())