DATA := /broad/compbio/aksarkar/projects/roadmap/data
HAPLOTYPES := $(DATA)/1kg
ENH := $(DATA)/roadmap/core-features/Enh
ENHCLUSTERS := $(DATA)/roadmap/hb-features/EnhClusters
FLAGS := --test 1000 --validation 1000
TRIALS := 50
EVALUATE := ctra-evaluate $(FLAGS)
export

SUBDIRS := gaussian-idealized-one-component \
	logistic-idealized-one-component \
	gaussian-idealized-two-component \
	logistic-idealized-two-component \
	gaussian-realistic-one-component \
	gaussian-realistic-synthetic-annotations \
	gaussian-realistic-Enh \
	gaussian-realistic-EnhClusters \
	gaussian-realistic-two-component \

.PHONY: $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

chr22.hdf5: pre
	ctra-convert -G {58C,NBS,BD,CAD,CD,HT,RA,T1D,T2D}{-clean.sample,_22.gen.gz} -n 16162 -p 129877 -o $@

pre:
	parallel -X ln -sf {} . ::: $(HAPLOTYPES:haplotypes=1kg-phase1)/*.{gen.gz,sample}
	touch $@

EnhClusters.txt.gz: $(notdir $(patsubst %.bed.gz,%.txt,$(wildcard $(ENHCLUSTERS)/*.bed.gz)))
	paste -d" " $^ | gzip >$@

%.txt: $(ENHCLUSTERS)/%.bed.gz
	zcat 58C_22.gen.gz | awk -vOFS='\t' '{delta = length($$5) - length($$4); if (delta >= 0) {print "chr22", $$3, $$3 + delta + 1} else {print "chr22", $$3 + length($$5), $$3 - delta + 1}}' | bedtools intersect -sorted -nobuf -c -a stdin -b $< | cut -f4 >$@

Enh.txt.gz: $(notdir $(patsubst %.bed.gz,%.txt,$(wildcard $(ENH)/*.bed.gz)))
	paste -d" " $^ | gzip >$@

%.txt: $(ENH)/%.bed.gz
	zcat 58C_22.gen.gz | awk -vOFS='\t' '{delta = length($$5) - length($$4); if (delta >= 0) {print "chr22", $$3, $$3 + delta + 1} else {print "chr22", $$3 + length($$5), $$3 - delta + 1}}' | bedtools intersect -sorted -nobuf -c -a stdin -b $< | cut -f4 >$@

synthetic-annotations: $(foreach i,1kb 1mb,synthetic-annotation-$(i).txt)

.PHONY: synthetic-annotations

synthetic-annotation-1kb.txt:
	zcat 58C_22.gen.gz | awk -vOFS='\t' '{if (int($$3 / 1e3) % 2) {print 0} else {print 1}}' >$@

synthetic-annotation-1mb.txt:
	zcat 58C_22.gen.gz | awk -vOFS='\t' '{if (int($$3 / 1e6) % 2) {print 0} else {print 1}}' >$@

.DELETE_ON_ERROR:
