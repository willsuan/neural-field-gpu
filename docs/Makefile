# Build the project writeup to a PDF.
#
# Targets:
#   make                   - build docs/writeup.pdf (and copy to repo root)
#   make figures           - regenerate figures from the benchmark CSVs
#   make clean             - remove aux files
#   make distclean         - remove generated PDFs too

TEX     = writeup.tex
BIB     = writeup.bib
NAME    = writeup
AUX     = $(NAME).aux $(NAME).log $(NAME).out $(NAME).bbl $(NAME).blg $(NAME).toc

ROOT    = ..
CSV_A   = $(ROOT)/bench_size.csv
CSV_B   = $(ROOT)/bench_gpu.csv
PLOT    = $(ROOT)/scripts/plot_scaling.py
FIGDIR  = figures

FIGS    = $(FIGDIR)/throughput_vs_size.pdf \
          $(FIGDIR)/time_per_step_vs_size.pdf \
          $(FIGDIR)/multi_gpu_scaling.pdf

.PHONY: all pdf figures clean distclean

all: pdf

pdf: $(NAME).pdf
	@cp $(NAME).pdf $(ROOT)/$(NAME).pdf
	@echo "wrote $(ROOT)/$(NAME).pdf"

$(NAME).pdf: $(TEX) $(BIB) $(FIGS)
	pdflatex -interaction=nonstopmode $(TEX) || true
	bibtex $(NAME) || true
	pdflatex -interaction=nonstopmode $(TEX) || true
	pdflatex -interaction=nonstopmode $(TEX)

figures: $(FIGS)

$(FIGS): $(CSV_A) $(CSV_B) $(PLOT)
	python3 $(PLOT)

clean:
	rm -f $(AUX)

distclean: clean
	rm -f $(NAME).pdf $(ROOT)/$(NAME).pdf
	rm -f $(FIGS)
