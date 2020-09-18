import requests
import random
import time


def splitRepoName(repoShortName: str):
    split = repoShortName.split("/")
    return split[0], split[1]


def buildNextPageQuery(cursor: str):
    if cursor is None:
        return ""
    return ', after:"{0}"'.format(cursor)


def runGraphqlRequest(pat: str, query: str):
    headers = {"Authorization": "Bearer {0}".format(pat)}

    sleepTime = random.randint(5, 15)
    time.sleep(sleepTime)

    request = requests.post(
        "https://api.github.com/graphql", json={"query": query}, headers=headers
    )

    if request.status_code == 200:
        return request.json()["data"]

    raise Exception(
        "Query execution failed with code {0}: {1}".format(
            request.status_code, request.text
        )
    )


def tryAddLogin(node, people: set):
    login = extractAuthorLogin(node)

    if not login is None:
        people.add(login)
        return True

    return False


def extractAuthorLogin(node):
    if node is None or not "login" in node or node["login"] is None:
        return None

    return node["login"]
