import requests


def countIssuesPerRepository(pat: str, owner: str, name: str):
    query = """{{
        repository(owner:"{0}", name:"{1}") {{
            issues {{
                totalCount
            }}
        }}
    }}""".format(
        owner, name
    )

    result = runGraphqlRequest(pat, query)

    totalCount = result["repository"]["issues"]["totalCount"]
    return totalCount


def countPullRequestsPerRepository(pat: str, owner: str, name: str):
    query = """{{
        repository(owner: "{0}", name: "{1}") {{
            pullRequests{{
                totalCount
            }}
        }}
    }}""".format(
        owner, name
    )

    result = runGraphqlRequest(pat, query)

    totalCount = result["repository"]["pullRequests"]["totalCount"]
    return totalCount


def countCommitsPerPullRequest(pat: str, owner: str, name: str):
    query = buildCountCommitsPerPullRequestQuery(owner, name, None)

    prCommitCounts = {}
    while True:

        # get page of PRs
        result = runGraphqlRequest(pat, query)

        # extract nodes
        nodes = result["repository"]["pullRequests"]["nodes"]

        # add results to dict
        for pr in nodes:
            prNumber = pr["number"]
            commitCount = pr["commits"]["totalCount"]
            prCommitCounts[prNumber] = commitCount

        # check for next page
        pageInfo = result["repository"]["pullRequests"]["pageInfo"]
        if not pageInfo["hasNextPage"]:
            break

        cursor = pageInfo["endCursor"]
        query = buildCountCommitsPerPullRequestQuery(owner, name, cursor)

    return prCommitCounts


def buildCountCommitsPerPullRequestQuery(owner: str, name: str, cursor: str):
    return """{{
        repository(owner: "{0}", name: "{1}") {{
            pullRequests(first:100{2}){{
                nodes {{
                    number
                    commits {{
                        totalCount
                        }}
                }}
                pageInfo {{
                    hasNextPage
                    endCursor
                }}
            }}
        }}
    }}""".format(
        owner, name, buildNextPageQuery(cursor)
    )


def getIssueParticipants(pat: str, owner: str, name: str):
    query = buildGetIssueParticipantsQuery(owner, name, None)

    participants = set()
    participantCount = dict()

    while True:

        # get page of PRs
        result = runGraphqlRequest(pat, query)

        # extract nodes
        nodes = result["repository"]["issues"]["nodes"]

        # add participants to list
        for node in nodes:
            count = 0

            # author
            if tryAddAuthorLogin(node["author"], participants):
                count += 1

            # editor
            if tryAddAuthorLogin(node["editor"], participants):
                count += 1

            # assignees
            for user in node["assignees"]["nodes"]:
                if tryAddAuthorLogin(user, participants):
                    count += 1

            # participants
            for user in node["participants"]["nodes"]:
                if tryAddAuthorLogin(user, participants):
                    count += 1

            participantCount[node["number"]] = count

        # check for next page
        pageInfo = result["repository"]["issues"]["pageInfo"]
        if not pageInfo["hasNextPage"]:
            break

        cursor = pageInfo["endCursor"]
        query = buildGetIssueParticipantsQuery(owner, name, cursor)

    return participants, participantCount


def buildGetIssueParticipantsQuery(owner: str, name: str, cursor: str):
    return """{{
        repository(owner: "{0}", name: "{1}") {{
            issues(first: 100{2}){{
                pageInfo {{
                    hasNextPage
                    endCursor
                }}
                nodes {{
                    number
                    author {{
                        ... on User {{
                            login
                        }}
                    }}
                    editor {{
                        ... on User {{
                            login
                        }}
                    }}
                    assignees(first: 100) {{
                        nodes {{
                            login
                        }}
                    }}
                    participants(first: 100) {{
                        nodes {{
                            login
                        }}
                    }}
                }}
            }}
        }}
    }}""".format(
        owner, name, buildNextPageQuery(cursor)
    )


def getPullRequestParticipants(pat: str, owner: str, name: str):
    query = buildGetPullRequestParticipantsQuery(owner, name, None)

    participants = set()
    participantCount = dict()

    while True:

        # get page of PRs
        result = runGraphqlRequest(pat, query)

        # extract nodes
        nodes = result["repository"]["pullRequests"]["nodes"]

        # add participants to list
        for node in nodes:
            count = 0

            # author
            if tryAddAuthorLogin(node["author"], participants):
                count += 1

            # editor
            if tryAddAuthorLogin(node["editor"], participants):
                count += 1

            # assignees
            for user in node["assignees"]["nodes"]:
                if tryAddAuthorLogin(user, participants):
                    count += 1

            # participants
            for user in node["participants"]["nodes"]:
                if tryAddAuthorLogin(user, participants):
                    count += 1

            participantCount[node["number"]] = count

        # check for next page
        pageInfo = result["repository"]["pullRequests"]["pageInfo"]
        if not pageInfo["hasNextPage"]:
            break

        cursor = pageInfo["endCursor"]
        query = buildGetPullRequestParticipantsQuery(owner, name, cursor)

    return participants, participantCount


def buildGetPullRequestParticipantsQuery(owner: str, name: str, cursor: str):
    return """{{
        repository(owner: "{0}", name: "{1}") {{
            pullRequests(first: 100{2}){{
                pageInfo {{
                    hasNextPage
                    endCursor
                }}
                nodes {{
                    number
                    author {{
                        ... on User {{
                            login
                        }}
                    }}
                    editor {{
                        ... on User {{
                            login
                        }}
                    }}
                    assignees(first: 100) {{
                        nodes {{
                            login
                        }}
                    }}
                    participants(first: 100) {{
                        nodes {{
                            login
                        }}
                    }}
                }}
            }}
        }}
    }}""".format(
        owner, name, buildNextPageQuery(cursor)
    )


def getIssueComments(pat: str, owner: str, name: str):
    issueCommentCount = dict()
    cursor = None

    while True:

        # get page of PRs
        query = buildGetIssueCommentsQuery(owner, name, cursor)
        result = runGraphqlRequest(pat, query)

        # extract nodes
        nodes = result["repository"]["issues"]["nodes"]

        # get count
        for node in nodes:
            issueCommentCount[node["number"]] = node["comments"]["totalCount"]

        # check for next page
        pageInfo = result["repository"]["issues"]["pageInfo"]
        if not pageInfo["hasNextPage"]:
            break

        cursor = pageInfo["endCursor"]

    return issueCommentCount


def buildGetIssueCommentsQuery(owner: str, name: str, cursor: str):
    return """{{
        repository(owner: "{0}", name: "{1}") {{
            issues(first: 100{2}) {{
                pageInfo {{
                    hasNextPage
                    endCursor
                }}
                nodes {{
                    number
                    comments {{
                        totalCount
                    }}
                }}
            }}
        }}
    }}""".format(
        owner, name, buildNextPageQuery(cursor)
    )


def buildNextPageQuery(cursor: str):
    if cursor is None:
        return ""
    return ', after:"{0}"'.format(cursor)


def runGraphqlRequest(pat: str, query: str):
    headers = {"Authorization": "Bearer {0}".format(pat)}

    request = requests.post(
        "https://api.github.com/graphql", json={"query": query}, headers=headers
    )
    if request.status_code == 200:
        return request.json()["data"]
    raise "Query execution failed with code {0}: {1}".format(
        request.status_code, request.text
    )


def tryAddAuthorLogin(node, list: set):
    login = extractAuthorLogin(node)

    if not login is None:
        list.add(login)
        return True

    return False


def extractAuthorLogin(node):
    if node is None or not "login" in node or node["login"] is None:
        return None

    return node["login"]
