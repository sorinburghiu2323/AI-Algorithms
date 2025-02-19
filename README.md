# AI-Algorithms

ECM2423 Coursework Exercise.

This code is hosted on github at: https://github.com/sorinburghiu2323/AI-Algorithms

## Setup

Please note that wanting to run just the 8-puzzle requires no setup.

First set up you python virtual environment and activate it.

```bash
$ python -m venv .venv

$ .venv\Scripts\activate.bat      # Windows
$ .venv\bin\activate              # Linux
```

Then install python dependencies:
```
$ pip install -r requirements.txt 
```

## A star algorithm: 8-puzzle

The python file `8-puzzle.py` contains the solution to the problem along with a nice cmd interface for the user.

To run it, simply run the python script. Development variables can be found at the top of the script and adjusted accordingly.

## K-means algorithm: hand-written digits 

Run the python file `k-means_clustering.py` having gone through the setup. The code will print out a table of data
in the console and a matplotlib interpretation of clusters should show up on the screen.

## Decision tree classifier: classification of urban areas

The python file `decision_trees.py` need the `data.csv` file to work. Make sure you have that installed
then you should be able to run the python file accordingly; setup step is also needed for this part.

Constants have been left at the top of the script to be adjusted if needed. The result should be a printout of the
decision tree accuracy and a generated graph to showcase feature importance.
