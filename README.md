# MIDI-quantisation

This version is developed for MIDI quantisation for classical piano music.

## MIDI quantisation using the pre-trained model

    from quantmidi import QuantMIDIProcessor
    processor = QuantMIDIProcessor(device='cuda')  # 'cuda' or 'cpu'
    processor.process('performance.mid', 'quantized.mid')

## Train a MIDI quantisation system from scratch

### 1. Download datasets

Download datasets...

### 2. Feature preparation

### 3. Training