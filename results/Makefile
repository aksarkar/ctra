HAPLOTYPES := /broad/compbio/aksarkar/projects/roadmap/data/1kg
FLAGS := --test 1000 --validation 1000
TRIALS := 50
EVALUATE := ctra-evaluate $(FLAGS)
export

SUBDIRS := gaussian-idealized-one-component \
	logistic-idealized-one-component \
	gaussian-idealized-two-component \
	logistic-idealized-two-component \
	gaussian-realistic-one-component \
	gaussian-realistic-synthetic-annotations

.PHONY: $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

chr22.hdf5: pre
	ctra-convert -G {58C,NBS,BD,CAD,CD,HT,RA,T1D,T2D}{-clean.sample,_22.gen.gz} -n 16162 -p 129877 -o $@

pre:
	parallel -X ln -sf {} . ::: $(HAPLOTYPES:haplotypes=1kg-phase1)/*.{gen.gz,sample}
	touch $@

annotations.txt:
	zcat 58C_22.gen.gz | awk -vOFS='\t' '{if (int($$3 / 1e6) % 2) {print 0} else {print 1}}' >$@

.DELETE_ON_ERROR:
