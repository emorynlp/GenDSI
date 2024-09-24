
# Dialogue State Human Evaluation Task

## Introduction

The goal of this task is to evaluate the quality of dialogue state representations using human judgements. Dialogue states are sets of slot-value pairs that contain key information about the dialogue:

```text
Customer: Hello, I'm looking for a flight from London to Paris.
Agent: Sure, when would you like to depart?
Customer: I'd like to leave on the 20th of July. Do you have any evening flights?

=>

    departure location: London
    destination: Paris
    departure date: 20th of July
    time of departure: evening
    flight available: ?
```

For this project, we are actually interested in the dialogue state UPDATE, which is the set of slot-value pairs that have been added to the dialogue state since the last turn. For example, in the above dialogue, the dialogue state update for the last Customer turn would be:

```text
    departure date: 20th of July
    time of departure: evening
    flight available: ?
```

## Annotation Tool Launching

To complete this annotation task, you will use a python-based graphical annoation tool. To use the tool, download and unzip the `annotation` folder.

The data you will be annotating will come with the tool and is located in the annotation/data folder as a .json file-- do not modify the content of any of these files directly. 

**Move the single .json file in the annotation/data folder with your name into the annotation/annotation folder to annotate that .json file**. Then, launch the tool like: 

```bash
cd annotation
python3 -m main.py
```

**Make sure to use python 3.10 or higher**

If you don't have python 3.10 as your default python version, you can install anaconda or miniconda (https://docs.conda.io/en/latest/miniconda.html), create a new environment with python 3.10, activate the environment, and then run the tool:

```bash
conda create -n py310 python=3.10
conda activate py310
cd annotation
python -m main.py
```

(Ask ChatGPT for help if you don't have conda/miniconda and need help installing it)

Once run, the tool should open in a new window, with instructions provided.

## Completing Tasks

When completing evaluations using the tool, please keep the following in mind:

**Timing**: For scientific reproduction and workload managment purposes, please start a timer when you begin annotating and stop/pause it whenever you take a break. Use a text file in the `annotation` folder to take a quick note of how long it takes you to complete the annotations, just so we have a semi-formal record of it.

**Quality**: Please do your best to annotate accurately, as this will help us get accurate estimates of the characteristics of various dialogue state data. You can go as fast as you want as long as you maintain reasonably clear decision making for each annotation-- just try to find a good rhythm, and feel free to ask us what to do if certain types of examples are consistently difficult to annotate.

**Blindness**: We can't reveal where exactly the data comes from, and you might notice that the data source is more or less anonymized-- this is intentional to avoid bias. While we might try to keep you in the dark for now about some project details, we'll be happy to reveal everything once all the data is annotated.  

**Communication**: I'll probably communicate with you to ask how you are doing anyway, but feel free to reach out via Teams text at any time if you have any questions or concerns!

**Delivery**: When you're done you can send .json file(s) in the annotation/annotation folder to me on Teams for analysis. 