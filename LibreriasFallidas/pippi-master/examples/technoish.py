from pippi import dsp, oscs, tune, rhythm, wavetables
import random

osc = oscs.Osc()
out = dsp.buffer(1)
kick = dsp.read('sounds/909kick.wav')
hat = dsp.read('sounds/hat.wav')
snare = dsp.read('sounds/606snare.wav')

numbars = 64 * 4
chords = 'ii ii ii V9'.split(' ')
chords2 = 'i i6 IV IV'.split(' ')

def rush(snd):
    out = dsp.buffer()
    length = random.randint(4410, 44100 * 3)
    numbeats = random.randint(16, 64)
    reverse = random.choice([True, False])
    wintype = random.choice(['sine', 'tri', 'kaiser', 'hann', 'blackman', None])
    wavetable = None if wintype is not None else [ random.random() for _ in range(random.randint(3, 10)) ]
    pattern = rhythm.curve(numbeats=numbeats, wintype=wintype, length=length, reverse=reverse, wavetable=wavetable)
    minspeed = random.triangular(0.15, 2)

    pan = random.random()

    for onset in pattern:
        hit = snd * random.triangular(0, 0.5)
        hit = hit.speed(random.triangular(minspeed, minspeed + 0.08))
        hit = hit.pan(pan)
        out.dub(hit, onset)

    return out

for i in range(numbars):
    bar = dsp.buffer()

    numlayers = random.randint(3, 6)
    beat = int(((60000 / 109)/1000) * 44100) // 4

    if i//4 % 3 == 0:
        chord = chords2[i%len(chords2)]
    else:
        chord = chords[i%len(chords)]

    freqs = tune.chord(chord, octave=3, key='d')

    if i > 3:
        for _ in range(numlayers):
            pattern = rhythm.pattern(16, random.randint(2,6), random.randint(0, 3))
            onsets = rhythm.onsets(pattern, beat)
            for onset in onsets:
                if random.random() > 0.5:
                    osc.freq = random.choice(freqs)
                    osc.wavetable = [0] + [ random.triangular(-1, 1) for _ in range(random.randint(3, 10)) ] + [0]
                    osc.amp = 0.5
                    snd = osc.play(random.randint(441, 44100))
                    snd = snd.env('phasor')
                    snd = snd.pan(random.random()) * random.triangular(0.1, 0.25)

                    bar.dub(snd, onset)

        snare_onsets = rhythm.onsets('.x.x', beat * 4)
        for onset in snare_onsets:
            s = snare.speed(random.triangular(0.9, 1.1)) * random.triangular(0.85, 1)
            bar.dub(s, onset)

            if random.random() > 0.75:
                wtypes = ['phasor', 'line', 'sine', 'tri']
                s = rush(s)
                try:
                    s = s.env(random.choice(wtypes)) * random.triangular(0.5, 1)
                except IndexError as e:
                    print(e)

                if len(s) > 0:
                    bar.dub(s, onset)
                else:
                    print(len(s), onset)

    if i//4 % 3 == 0:
        hat_onsets = rhythm.onsets([1,1,0,0] * 4, beat)
        for onset in hat_onsets:
            h = hat.copy()
            h.fill(441*random.randint(10, 30))
            h = h.env('phasor').speed(random.triangular(1, 1.25))
            bar.dub(h * random.triangular(0.25, 0.3), onset)

    kick_onsets = rhythm.onsets([1, 0, 0, 0], beat * 4)
    for onset in kick_onsets:
        k = kick.copy()
        k.fill(441*random.randint(10, 60))
        k = k.env('phasor').speed(random.triangular(1.4, 1.6))
        bar.dub(k * random.triangular(0.25, 0.35), onset)

    osc.freq = freqs[0] * 0.25
    bass_onsets = rhythm.onsets([0,0,1] * 3, beat * 3)
    for onset in bass_onsets:
        osc.wavetable = wavetables.wavetable('square', 512)
        bass = osc.play(beat * random.randint(1,3)).env('phasor') * 0.08
        bar.dub(bass, onset)

        osc.wavetable = wavetables.wavetable('tri', 512)
        bass = osc.play(beat * random.randint(2,5)).env('phasor') * 0.35
        bar.dub(bass, onset)
        
    bar.fill(beat * 16)
    out += bar

out.write('technoish.wav')
