alias r := run
alias s := shell
alias id := install_dep
alias ud := update_dep

run:
  poetry run python3 src/sentimaniac/main.py

shell:
  poetry shell

install_dep:
  python3 -m pip install pipx
  pipx install poetry
  poetry install

update_dep:
  poetry update
  
