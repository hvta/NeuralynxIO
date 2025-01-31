# NeuralynxIO

Python script for importing Neuralynx data files. Currently only .NCS (continuously sampled record) and .NEV (event 
record) files are supported. The data is extracted from the binary files and placed in Numpy arrays and native Python
types.

This fork includes Python's memmap option so large files can also be read without loading everything into memory.

## File formats

Neuralynx provides PDF documentation of their file formats [online](http://neuralynx.com/software/NeuralynxDataFileFormats.pdf)
(although there are some discrepancies with the files actually generated by their software).

## Use

    import neuralynxio
  
    ncs = neuralynxio.load_ncs('./Chan1.ncs')  # Load signal data into a dictionary
    ncs['data'][0]  # Access the first sample of data
  
    nev = neuralynxio.load_nev('./Event.nev')  # Load event data into a dictionary
    nev['event'][0]  # Access the first event
