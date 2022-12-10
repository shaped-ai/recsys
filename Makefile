# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=1

clean:
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns")
	rm -rf _ckpt_*
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf tests/.pytest_cache
	rm -rf ./docs/build
	rm -rf ./docs/source/generated
	rm -rf ./docs/source/*/generated
	rm -rf ./docs/source/api
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf src/*.egg-info

docs: clean
	pip install -e .
	pip install --quiet -r docs/requirements.txt
	sphinx-apidoc -f --implicit-namespaces -o docs/source recsys/
	python -m sphinx -b html -W --keep-going docs/source docs/build