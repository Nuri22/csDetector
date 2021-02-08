# create virtual environment to avoid polluting global namespace
py -m venv .venv

# activate environment
.venv/Scripts./Activate.ps1

# install modules
pip install -r requirements.txt

# install convokit dependencies
py -m spacy download en_core_web_sm