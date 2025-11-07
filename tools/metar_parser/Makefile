DESTDIR := ../../src/metpy/io

PYMOD := $(DESTDIR)/_metar_parser.py

all: $(PYMOD)

clean:
	rm -f $(PYMOD)

canopy:
	npm install -g canopy

$(DESTDIR)/_%.py: %.peg
	canopy $? --lang python --output $(basename $@)
