import os
import git

from configuration import Configuration


def getRepo(config: Configuration):

    # build path
    repoPath = os.path.join(
        config.repositoryPath,
        "{}.{}".format(config.repositoryOwner, config.repositoryName),
    )

    # get repository reference
    repo = None
    if not os.path.exists(repoPath):
        print("Downloading repository...")
        repo = git.Repo.clone_from(
            config.repositoryUrl,
            repoPath,
            branch="master",
            progress=Progress(),
            odbt=git.GitCmdObjectDB,
        )
        print()
    else:
        repo = git.Repo(repoPath, odbt=git.GitCmdObjectDB)

    return repo


class Progress(git.remote.RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=""):
        print(self._cur_line, end="\r")