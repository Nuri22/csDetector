import git

def authorIdExtractor(author: git.Actor):
    id = ""
    
    if author.email is None:
        id = author.name
    else:
        id = author.email
        
    id = id.lower().strip()
    return id

def iterLen(obj: iter):
    return sum(1 for _ in obj)