# General Notes

## Tool Overview

The purpose of this tool is twofold:

1. Contribute to the list of existing community smells that developers need to be aware of in their community.
2. Provide developers with a tool to automatically detect community smells in their project. 

## Stack
- Windows 10
- VSCode 1.45.1
- PowerShell 7.0.1
- Python 3.8.3

# Installation
All required modules **must be installed** prior to running the tool.

## Recommended
Run **installModules.ps1** in PowerShell for a quick and simple setup. This will create an embedded **venv** environment and install all the necessary modules with correct versions without polluting the global namespace.

## Manual
Inspect **installModules.ps1** and **requiredModules.txt** files for a manual installation in an environment that does not have PowerShell.

# Running
If you followed the recommended installation approach, the virtual environment **must be activated** prior to running the tool! Run the proper to your environment activation script inside the **.venv/Scripts** folder.

To run the tool you need run **devNetwork.py** with the right parameters. Pass the **--help** parameter to view the documentation.

## Configuration File

#### aliasSimilarityMaxDistance (float)
For documentation on changing this value see:  
https://github.com/luozhouyang/python-string-similarity#metric-longest-common-subsequence  
*Ex: 0.75*

#### The Sentistrength tool
The files of sentistrength tool are available here: 
http://sentistrength.wlv.ac.uk/jkpop/ 

# Aliases
It is recommended to generate and massage author aliases prior to analyzing repositories to minimize the number of duplicate users who have historically used multiple emails for their commits skewing the developer network analysis.

To generate author aliases, run **authorAliasExtractor.py** with the right parameters. Pass **--help** for parameter documentation.