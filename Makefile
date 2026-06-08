# ============================================================
#  铁电相变加速计算 - Makefile
# ============================================================

SHELL := /bin/bash

.PHONY: help install compile-qe run clean

help:
	@echo "=============================================="
	@echo "  铁电相变加速计算"
	@echo "  ALIGNN-FF (GPU) + QE (GPU)"
	@echo "=============================================="
	@echo ""
	@echo "  make install     - Install all Python dependencies"
	@echo "  make compile-qe  - Compile QE with GPU support"
	@echo "  make run         - Run pipeline (needs INPUT set)"
	@echo "  make run ARGS='my.in -o results'"
	@echo "  make clean       - Clean output files"
	@echo ""
	@echo "  Examples:"
	@echo "    make run ARGS='paraelectric.in'"
	@echo "    make run ARGS='paraelectric.in -o my_out --no-qe'"
	@echo "    make run ARGS='structure.POSCAR -a 0.15 -n 15'"

install:
	pip install ase jarvis-tools pandas scipy
	cd alignn-main && pip install -e .

compile-qe:
	@echo "Compiling QE with GPU support..."
	cd q-e-develop && \
		./configure --with-gpu=cuda --enable-openmp && \
		make pw
	@echo "QE compiled. pw.x is in q-e-develop/bin/"

run:
	@if [ -z "$(INPUT)" ] && [ -z "$(ARGS)" ]; then \
		echo "Usage: make run ARGS='<input_file> [options]'"; \
		echo "   or: make run INPUT='paraelectric.in'"; \
		exit 1; \
	fi
	python3 -m ferro_pipeline.cli $(INPUT) $(ARGS)

clean:
	rm -rf ferro_results*/
	@echo "Cleaned output directories"

# Default target
all: install run
