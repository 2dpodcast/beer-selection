SHELL := /bin/bash

BA = 'http://snap.stanford.edu/data/beeradvocate.txt.gz'
RB = 'http://snap.stanford.edu/data/ratebeer.txt.gz'

DATA := beer-data.txt
SCRIPT := beer-selection.py
FIGURES := *.svg

.PHONY: all clean distclean

all: $(FIGURES)

$(FIGURES): $(SCRIPT) $(DATA)
	@echo Running analysis ...
	python $(SCRIPT) < $(DATA)

$(DATA):
	@echo Downloading data ...
	wget -qO- $(BA) $(RB) | gunzip -c > $@
	touch $@

clean:
	@- $(RM) $(DATA) $(FIGURES)

distclean: clean
