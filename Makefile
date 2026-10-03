.PHONY: serve setup

serve:
	.venv/bin/python harness_server.py

setup:
	brew bundle --no-upgrade
	brew bundle exec -- python3.14 -m venv .venv
	.venv/bin/python -m pip install -r requirements-server.txt
