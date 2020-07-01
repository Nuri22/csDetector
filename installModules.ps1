# create virtual environment to avoid polluting global namespace
py -m venv .venv

# activate environment
.venv/Scripts./Activate.ps1

# install modules
pip install -r requiredModules.txt