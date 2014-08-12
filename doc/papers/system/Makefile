                                                                                                                                                               
all: papers

clean:
	-rm -f *.aux *.log *.dvi *.blg *.out ms.pdf

# ----------------------------------------------------------------------------

pdf: ms.pdf

papers: pdf

# ----------------------------------------------------------------------------

# ms.bbl: ms.tex references.bib
# 	bubble -f ms.tex references.bib >& ms.bbl.log
# 	cat ms.bbl.log
      
ms.pdf: ms.tex ms.bbl
	pdflatex ms.tex

# ----------------------------------------------------------------------------

