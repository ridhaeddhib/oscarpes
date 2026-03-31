PYTHON      ?= python
SPHINXBUILD  = $(PYTHON) -m sphinx
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: help html clean

help:
	@echo "Usage: make html   — build HTML documentation"
	@echo "       make clean  — remove build directory"

html:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo ""
	@echo "Build finished. Open $(BUILDDIR)/html/index.html"

clean:
	rm -rf "$(BUILDDIR)"
