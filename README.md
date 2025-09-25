# SinesumPy

SinesumPy is a Python repo for visualizing sine wave summations. It is designed to help users understand Fourier series and signal processing concepts through interactive harmonic summations. It has the functionality to play the resultant summation or save as a WAV audio file.

## Authors

This is inspired from the Matlab script written for EE 261  taught by Brad Osgood at Stanford and ascribes the creation of that script as follows:

> The latest version of the application was written by Michelle Daniels, a student in EE261 back in Autumn 2006 (!), further modified by Daniel Kopeinigg in 2010, based on a clunky version written earlier by yours truly.

This is adapted by Abhishek Karve aided by GPT.

## Features

- Generate sine wave harmonic summations with customizable amplitudes and phases.
- Visualize waveforms.
- Play the waveform as sound or save as WAV audio file.
- Save/Load harmonic amplitudes and phases to/from JSON files.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/SinesumPy.git
cd SinesumPy
```

OR just download the zip file and run either of the .py files.

### Requirements
For full functionality just excute the follwoing from within the repo directory:

```bash
pip install requirements.txt
```

- ```sinesum_Matlab-Look.py``` only uses numpy and matplotlib to do the UI and the calculations.
  - tk, scipy, and soundevice can be installed for added functionality (for dialogue boxes and message boxes, saving wav files, and playing the audio respectively). Checks are in place, so not having these libraries is still fine if you just want to run the app.
- ```sinesum_pyqt.py``` uses pyqt for the UI and looks a bit more modern
  - scipy, and soundevice can be installed for added functionality (saving wav files and playing the audio respectively). Checks are in place, so not having these libraries is still fine of you just want to run the app.

## Usage

Run either of the ```.py``` files with:

```bash
python sinesum_Matlab-Look.py
```

OR

```bash
python sinesum_pyqt.py
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.
