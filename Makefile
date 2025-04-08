.PHONY: black test

PYTHON = python3

black:
	$(PYTHON) -m venv black_env; \
	bash -c "source black_env/bin/activate && \
		pip install --upgrade pip && \
		pip install black && \
		pip install -e . && \
		black . --line-length 120";
	rm -rf black_env

test:
	$(PYTHON) -m venv test_env; \
	bash -c "source test_env/bin/activate && \
		pip install --upgrade pip && \
		pip install pytest && \
		pip install -e . && \
		pytest tests/"
	rm -rf test_env
