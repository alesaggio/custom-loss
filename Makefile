install:
	pip3 install --upgrade pip &&\
		pip3 install -r requirements.txt

format:
	python3 -m black *.py

lint:
	python3 -m pylint --disable=R,C custom_loss.py

test:
	python3 -m pytest -vv test_custom_loss.py

all: install format lint test