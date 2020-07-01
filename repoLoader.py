import os
import git

from configuration import Configuration

def getRepo(config: Configuration):
    
    # get repository reference
    repo = None
    if not os.path.isdir(config.repositoryPath):
        print("Downloading repository...")
        repo = git.Repo.clone_from(
                config.repositoryUrl,
                config.repositoryPath,
                branch='master',
                progress=Progress(),
                odbt=git.GitCmdObjectDB)   
        print()
    else:
        repo = git.Repo(config.repositoryPath, odbt=git.GitCmdObjectDB)
        
    return repo

class Progress(git.remote.RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        print(self._cur_line, end="\r")