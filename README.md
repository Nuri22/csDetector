# General Notes

## Summary
This tool performs network and statistical analysis on GitHub repositories. The results can be used to detect community smells.

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

To run the tool you need to setup a configuration file **[something].yml** and run **main.py** with the right parameters:
- **-c** (**--config**): physical location of the configuration file
- **-p** (**--pat**): GitHub PAT (personal access token) used for querying the GitHub API

Your final result should look like this:  
**$ py main.py -c config.yml -p abc123456**

## Configuration File

#### repositoryUrl (str)
The URL to the GitHub repository you want to analyse.  
*Ex: "https://github.com/eclipse/rt.equinox.bundles"*

#### repositoryShortname (str)
The user/project part of the repository URL. Must be exact.  
*Ex: "eclipse/rt.equinox.bundles"*

#### repositoryPath (str)
The physical path to the local clone of the repository.
If the directory does not exist, it will be created and the repository will be automatically cloned to this path.   
*Ex: "D:\Repos\rt.equinox.bundles"*

#### analysisOutputPath (str)
The physical path to the desired output directory.  
*Ex: "D:\Repos\analysisOutput\eclipse-rt.equinox.bundles"*

#### aliasPath (str)
The physical path to the list of author aliases.  
*Ex: "D:\Repos\analysisOutput\aliases\eclipse-rt.equinox.bundles.yml"*

#### aliasSimilarityMaxDistance (float)
For documentation on changing this value see:  
https://github.com/luozhouyang/python-string-similarity#metric-longest-common-subsequence  
*Ex: 0.75*

# Aliases
It is recommended to generate and massage author aliases prior to analyzing repositories to minimize the number of duplicate users who have historically used multiple emails for their commits skewing the developer network analysis.

To generate the initial aliases, run the **aliasLoginJoiner.py** file followed by your PAT as the first parameter:  
**$ py aliasLoginJoiner.py abc123456**