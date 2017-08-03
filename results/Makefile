FLAGS := --test 1000 --validation 1000
TRIALS := 50
EVALUATE := ctra-evaluate $(FLAGS)
export

SUBDIRS := gaussian-idealized-one-component logistic-idealized-one-component gaussian-idealized-two-component logistic-idealized-two-component

.PHONY: $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

chr22.hdf5: pre
	ctra-convert -G {58C,NBS,BD,CAD,CD,HT,RA,T1D,T2D}{-clean.sample,_22.gen.gz} -n 16162 -p 129877 -o $@

pre:
	parallel -X ln -sf {} . ::: $(HAPLOTYPES:haplotypes=1kg-phase1)/*.{gen.gz,sample}
	touch $@

chr%:
	parallel --halt now,fail=1 -j1 gtool -S --g $(HAPLOTYPES)/{}_$*.lift.gz --s $(HAPLOTYPES)/{}.sample --og {}_$*.gen.gz --os {}_$*.sample --sample_excl $(HAPLOTYPES)/{}.exclude --exclusion $(HAPLOTYPES)/snps.exclude ::: 58C NBS BD CAD CD HT RA T1D T2D
	gunzip *.sample.gz
	touch $@

annotations.txt:
	zcat 58C_01.gen.gz | awk -vOFS='\t' '{print "chr"$$1, $$3, $$3 + length($$5) - length($$4)}' | bedtools intersect -sorted -nobuf -c -a stdin -b /broad/compbio/aksarkar/projects/roadmap/data/roadmap/core-features/Enh/E042.bed.gz | cut -f4 >$@

.DELETE_ON_ERROR:
