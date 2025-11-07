DESTDIR := ../../src/metpy/io/_metar_parser

PYMOD := $(DESTDIR)/metar_parser.py

all: $(PYMOD)

clean:
	rm -f $(PYMOD)

canopy:
	npm install -g canopy

$(DESTDIR)/%.py: %.peg
	canopy $? --lang python --output $(basename $@)
