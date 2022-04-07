import pretty_midi as pm
from functools import reduce, cmp_to_key
from pathlib import Path
import pandas as pd



class DataUtils():
    
    @staticmethod
    def get_note_sequence_from_midi(midi_file):
        """
        Get note sequence from midi file.
        """
        midi_data = pm.PrettyMIDI(str(Path(midi_file)))
        note_sequence = reduce(lambda x, y: x+y, [inst.notes for inst in midi_data.instruments])
        note_sequence = sorted(note_sequence, key=cmp_to_key(DataUtils.compare_note_order))
        return note_sequence

    @staticmethod
    def get_note_sequence_and_beats_from_midi(midi_file):
        """
        Get beat sequence from midi file.
        """
        midi_data = pm.PrettyMIDI(str(Path(midi_file)))
        note_sequence = reduce(lambda x, y: x+y, [inst.notes for inst in midi_data.instruments])
        note_sequence = sorted(note_sequence, key=cmp_to_key(DataUtils.compare_note_order))
        beats = midi_data.get_beats()
        return note_sequence, beats

    @staticmethod
    def get_beats_from_annot_file(annot_file):
        """
        Get beat sequence from annotation file.
        """
        annot_data = pd.read_csv(str(Path(annot_file)), header=None, sep='\t')
        beats = annot_data[0].tolist()
        return beats

    @staticmethod
    def compare_note_order(note1, note2):
        """
        Compare two notes by firstly onset and then pitch.
        """
        if note1.start < note2.start:
            return -1
        elif note1.start == note2.start:
            if note1.pitch < note2.pitch:
                return -1
            elif note1.pitch == note2.pitch:
                return 0
            else:
                return 1
        else:
            return 1