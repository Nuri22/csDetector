class Configuration:
    def __init__(self, repositoryUrl: str, repositoryPath: str, repositoryShortname: str, analysisOutputPath: str, aliasPath:str, aliasSimilarityMaxDistance: float):
        self.repositoryUrl = repositoryUrl
        self.repositoryShortname = repositoryShortname
        self.repositoryPath = repositoryPath
        self.analysisOutputPath = analysisOutputPath
        self.aliasPath = aliasPath
        self.aliasSimilarityMaxDistance = aliasSimilarityMaxDistance