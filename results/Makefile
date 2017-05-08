DATA := /broad/compbio/aksarkar/projects/roadmap/data/1kg
FLAGS := --test 1000 --validation 1000
TRIALS := 50
EVALUATE := ctra-evaluate $(FLAGS)

export

all:
	$(MAKE) -C gaussian-idealized-one-component
	$(MAKE) -C logistic-idealized-one-component

chr22.hdf5: pre
	ctra-convert -G {58C,NBS,BD,CAD,CD,HT,RA,T1D,T2D}{-clean.sample,_22.gen.gz} -n 16162 -p 129877 -o $@

pre:
	parallel -X ln -sf {} . ::: $(DATA)/1kg-phase1/*.{gen.gz,sample}
	touch $@

.DELETE_ON_ERROR:
