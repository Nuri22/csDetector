# General Notes

## Tool Overview

The purpose of this tool is twofold:

1. Contribute to the list of existing community smells that developers need to be aware of in their community.
2. Provide developers with a tool to automatically detect community smells in their project.


 ## Video Demo
 
 
 [![Demo ](https://youtu.be/AarXmePrEXA)]


## There are two ways to run the tool: 

## 1) By stand-alone executable file under windows:
- Windows 10
- Python 3.8.3
- Java 8.0.231 
- Open the folder that contains the executable file “devNetwork.exe” in command prompt "c:\tool_path\dist\devNetwork\".
- run devNetwork.exe with the right parametres see the [Running](#Running) section.

## 2) By the command line:

## Stack
- Windows 10
- VSCode 1.45.1
- PowerShell 7.0.1
- Python 3.8.3
- Java 8.0.231

# Installation
All required modules **must be installed** prior to running the tool.

## Recommended
Run **installModules.ps1** in PowerShell for a quick and simple setup. This will create an embedded **venv** environment and install all the necessary modules with correct versions without polluting the global namespace.

## Manual

### Packages
Inspect **installModules.ps1** and **requiredModules.txt** files for a manual installation in an environment that does not have PowerShell.

## ConvoKit
The tool requires ConvoKit to be installed correctly. See the [references](#references) section for setup instructions.


# Running
If you followed the recommended installation approach, the virtual environment **must be activated** prior to running the tool! Run the proper to your environment activation script inside the **.venv/Scripts** folder.

To run the tool you need run **devNetwork.py** with the right parameters. Pass the **--help** parameter to view the documentation. For example:
- (-p) for "GitHub PAT (personal access token) used for querying the GitHub API". 
- **Optional**(-g) for "Google Cloud API Key used for authentication with the Perspective API". 
- (-r) for "GitHub repository URL that you want to analyse". 
- (-s) for  "local directory path to the SentiStregth tool" See the [references](#references) section. 
- (-o) for "Local directory path for analysis output".
- **Optional**(-sd) for “The desired date to start analyzing a project  YYYY-MM-DD”.
## Configuration File

#### aliasSimilarityMaxDistance (float)
For documentation on changing this value see:  
https://github.com/luozhouyang/python-string-similarity#metric-longest-common-subsequence  
*Ex: 0.75*

# Aliases
It is recommended to generate and massage author aliases prior to analyzing repositories to minimize the number of duplicate users who have historically used multiple emails for their commits skewing the developer network analysis.

To generate author aliases, run **authorAliasExtractor.py** with the right parameters. Pass **--help** for parameter documentation.

# References
- GitHub GraphQL API Explorer  
https://docs.github.com/en/graphql/overview/explorer

-  The files of sentistrength tool are "SentiStrength_Data.zip & SentiStrength.jar" and available on: 
    http://sentistrength.wlv.ac.uk/jkpop/ 
	 - You need to extrcat the SentiStrength_Data.zip folder.
	 - You need to install Java in your local device to work with SentiStrength.jar file.
	 - (-s) parameter for "local directory path to the folder that contains these two files"

- ConvoKit
  - Setup  
  https://convokit.cornell.edu/documentation/tutorial.html

  - Politeness features and Markers in Convokit  
  https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/politeness-strategies/Politeness%20Marker%20and%20Summarize%20Demo.ipynb


### Metrics definitions

#### Developer Contributions metrics  

```NoD ```: Number of developers (NoD): the total number of developers who have changed the code in a project.

```NAD``` : Number of Active Days of an author on a project.

```NCD``` : Number of Commits per Developer in a project.

```SDC``` : Standard Deviation of Commits per developer in a project.

```NCD``` : Number of Core Developers.

```PCD``` : Percentage of Core Developers.

```NSD``` : Number of Sponsored Developers.

```PSD``` : Percentage of Sponsored Developers.

```NPR``` : Total number of Pull Requests.

```SAPR``` : Standard deviation of authors per PR.

```ANAPR``` : Average number of authors per PR.

```NI``` : Number of Issues.

```SDAI``` : Standard deviation of authors per issue report.

```ANAI``` : The average number of authors per issue report.

#### Social Network Analysis metrics

```GDC``` : Graph Degree Centrality.

```SDD``` : Standard Deviation of a graph Degree centrality in a project.

```GBC``` : Graph Betweenness Centrality.

```GCC``` : Graph Closeness Centrality.

```ND``` : Network Density.

```CC``` : Graph Closeness centrality

```ND``` : Network Density


#####  Community metrics     

```NC``` : Number of Communities.

```ACC``` : Average of Commits per Community.

```SCC``` : Standard deviation of Commits per Community.

```ADC``` : Average number of Developers per Community.

```SDC``` : Standard deviation of Developers per Community.


####  Geographic Dispersion metrics 

```TZ``` : Number of time zones

```ACZ``` : Average of Commits per time Zone.

```SCZ``` : Standard deviation of Commits per time Zones.

```ADZ``` : Average number of Developers per time Zone.

```SDZ``` : Standard deviation of Developers per time Zones.

####  Formality metrics 

```NR``` : Number of Releases in a project.

```PCR``` : Parentage of Commits per Release.

```FN``` : Formal Network.

```ADPR``` : Average number of days per PR.

```ADI``` : Average number of days per issue report.

####  Truck Number and Community Members metrics 

```BFN``` : Bus Factor Number.

```TFN``` : Global TruckNumber.

```TFC``` : Truck Factor Coverage.

####  Communication metrics

```ANCPR``` : Average number of comments per PR.

```SCPR``` : Standard deviation of commits per PR.

```NCI``` : Number Comments in issues.

```ANCI``` : Average number of comments per issue report.

```SDCI``` : Standard Deviation of Comments Count per issue report.

#### Sentiment Analysis metrics 

```RTCPR```: Ratio of toxic comments in PR discussions.

```RTCI``` : Ratio of toxic comments in issue discussions.

```RPCPR``` : Ratio of polite comments in PR discussions.

```RPCI``` : Ratio of polite comments in issue discussions.

```RINC``` : Ratio of issues with negative sentiments.

```RNSPRC``` : Ratio of negative sentiments in PR comments.

```RAWPR``` : Ratio of anger words in PR discussions.

```RAWI``` : Ratio of anger words in PR discussions.

```ACCL``` : Average Communication Comments Length.


