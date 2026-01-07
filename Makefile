.PHONY: all check format test

all: check format test

check:
	ruff check

format: 
	ruff format

test:
	pytest