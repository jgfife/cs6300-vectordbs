VENV := .virtual_environment

all: help

help:
	@echo
	@echo "Targets:"
	@echo "install                     - Install environment necessary to support this project."
	@echo "install-deb                 - Install OS packages necessary to support this project. Assumes apt/dpkg package management system."
	@echo "install-pip                 - Install Python pakcages necessary to suport this project."
	@echo "chroma-demo    			   - Run the Chroma demo."
	@echo "clean                       - Delete contents of db/chroma folder."
	@echo

$(VENV):
	python3 -m venv $(VENV)

install: install-deb install-pip

install-deb:
	@echo python3.12-venv is necessary for venv.
	@echo ffmpeg is necessary to read audio files for ASR
	for package in python3.12-venv ffmpeg; do \
		dpkg -l | egrep '^ii *'$${package}' ' 2>&1 > /dev/null || sudo apt install $${package}; \
	done

install-pip: $(VENV)
	. $(VENV)/bin/activate; pip3 install --upgrade -r requirements.txt

chroma: $(VENV)
	. $(VENV)/bin/activate; python3 src/chroma.py

pinecone: $(VENV)
	. $(VENV)/bin/activate; python3 src/pineconeDB.py

clean:
	rm -rf db/chroma/*