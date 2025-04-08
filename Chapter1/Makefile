clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.pdf *.toc *.nav *.snm *.vrb *.lof *.lot *.lol *.fls *.fdb_latexmk *.synctex.gz *.dvi *.ps *.spl *.backup *.bak *.synctex.gz\(busy\) *.synctex.gz\(busy\)
report:
	xelatex report.tex
	bibtex report
	xelatex report.tex
	xelatex report.tex
	