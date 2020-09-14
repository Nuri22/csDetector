class Configuration:
    def __init__(
        self,
        repositoryUrl: str,
        repositoryPath: str,
        batchSizeInMonths: int,
        repositoryShortname: str,
        analysisOutputPath: str,
        aliasPath: str,
        aliasSimilarityMaxDistance: float,
        sentiStrengthJarPath: str,
        sentiStrengthDataPath: str,
    ):
        self.repositoryUrl = repositoryUrl
        self.repositoryShortname = repositoryShortname
        self.repositoryPath = repositoryPath
        self.batchSizeInMonths = batchSizeInMonths
        self.analysisOutputPath = analysisOutputPath
        self.aliasPath = aliasPath
        self.aliasSimilarityMaxDistance = aliasSimilarityMaxDistance
        self.sentiStrengthJarPath = sentiStrengthJarPath
        self.sentiStrengthDataPath = sentiStrengthDataPath