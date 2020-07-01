import os
import git
import yaml

from typing import List
from progress.bar import Bar
from utils import authorIdExtractor

def replaceAliases(commits: List[git.Commit], aliasPath: str):
    print("Cleaning aliased authors")
    
    # quick lowercase and trim if no alias file
    if aliasPath == None or not os.path.exists(aliasPath):
        return replaceAll(commits, {})
    
    # read aliases
    content = ""
    with open(aliasPath, 'r', encoding='utf-8-sig') as file:
        content = file.read()
        
    aliases = yaml.load(content, Loader=yaml.FullLoader)
    
    # transpose for easy replacements
    transposesAliases = {}
    for alias in aliases:
        for email in aliases[alias]:
            transposesAliases[email] = alias
            
    # replace all author aliases with a unique one
    return replaceAll(commits, transposesAliases)

def replaceAll(commits, aliases):
    for commit in Bar('Processing').iter(list(commits)):
        copy = commit
        author = authorIdExtractor(commit.author)
        
        if author in aliases:
            copy.author.email = aliases[author]
            yield copy
        else:
            copy.author.email = author
            yield commit