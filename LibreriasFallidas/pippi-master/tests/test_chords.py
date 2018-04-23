from unittest import TestCase

from pippi import tune

class TestChords(TestCase):
    def test_getQuality(self):
        assert tune.getQuality('ii') == '-'
        assert tune.getQuality('II') == '^'
        assert tune.getQuality('vi69') == '-'
        assert tune.getQuality('vi6/9') == '-'
        assert tune.getQuality('ii7') == '-'
        assert tune.getQuality('v*9') == '*'

    def test_getExtension(self):
        assert tune.getExtension('ii') == ''
        assert tune.getExtension('II') == ''
        assert tune.getExtension('vi69') == '69'
        assert tune.getExtension('vi6/9') == '69'
        assert tune.getExtension('ii7') == '7'
        assert tune.getExtension('v*9') == '9'

    def test_getIntervals(self):
        assert tune.getIntervals('ii') == ['P1', 'm3', 'P5']
        assert tune.getIntervals('II') == ['P1', 'M3', 'P5']
        assert tune.getIntervals('II7') == ['P1', 'M3', 'P5', 'm7']
        assert tune.getIntervals('v6/9') == ['P1', 'm3', 'P5', 'M6', 'M9']

    def test_addIntervals(self):
        assert tune.addIntervals('P5','P8') == 'P12'
        assert tune.addIntervals('m3','P8') == 'm10'
        assert tune.addIntervals('m3','m3') == 'TT'
        assert tune.addIntervals('m3','M3') == 'P5'

    def test_getChord(self):
        assert tune.chord('I7', key='a', octave=4, ratios=tune.just) == [440.0, 550.0, 660.0, 792.0] 
        assert tune.chord('I7', key='a', octave=3, ratios=tune.just) == [220.0, 275.0, 330.0, 396.0] 


    def test_getRatiofromInterval(self):
        assert tune.getRatioFromInterval('P1', tune.just) == 1.0
        assert tune.getRatioFromInterval('P5', tune.just) == 1.5
        assert tune.getRatioFromInterval('P8', tune.just) == 2.0
        assert tune.getRatioFromInterval('m10', tune.just) == 2.4
        assert tune.getRatioFromInterval('P15', tune.just) == 4.0
