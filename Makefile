UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	ifeq ($(USE_GPU),true)
        PIP_REQUIREMENTS := requirements-linux-gpu.txt
    else
        PIP_REQUIREMENTS := requirements-linux-cpu.txt
    endif
endif
ifeq ($(UNAME_S),Darwin)
	PIP_REQUIREMENTS := requirements-osx.txt
endif

init:
	pip install --no-cache-dir -r $(PIP_REQUIREMENTS)

test:
	nosetests tests

test-coverage:
	nosetests --with-coverage --cover-package=agent --cover-xml tests

install:
	python setup.py install

train-new:
	python -m agent train-new

train-resume:
	python -m agent train-resume -s ${SESSION_ID}

play:
	python -m agent play -s ${SESSION_ID}

visualize-tsne:
	python -m agent visualize-tsne -s ${SESSION_ID}

metrics-show:
	python -m agent metrics-show -s ${SESSION_ID}

metrics-export:
	python -m agent metrics-export -s ${SESSION_ID}