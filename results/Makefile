HAPLOTYPES := /broad/hptmp/aksarkar/haplotypes
SAMPLE := -n 1000 -p 1000
GENO :=
MODEL := -m gaussian --center
METHOD := -O wsabi -M coord
PVE := --true-pve
EMPTY :=
BOOTSTRAP := parametric-bootstrap
FLAGS :=

DATA := $(SAMPLE) $(GENO)
EVALUATE := ctra-evaluate $(FLAGS)

%.txt.gz:
	awk 'FNR == 1 {sub("./$*-", "", FILENAME); sub(".txt", "", FILENAME); gsub("-", " ", FILENAME)} !/WARNING/ {print FILENAME, FNR, $$1}' ./$*-*.txt | sort -k1,4n | gzip >$@

define ONE_COMPONENT
BEGIN {
    for (logp=3; logp < 5; logp += 1) {
        for (np=.25; np <= 1; np += 0.25) {
            for (seed=0; seed < 10; seed += 1) {
                m=int(10 ^ (logp - 2));
                p = 10^logp; n=np * p;
		if (n < 16120 || "$(GENO)" == "") {
	            sink = ">$(word 2, $(METHOD))-$(word 2, $(MODEL))-sample-size-%d-%d-%d.txt\n"
                    command = "$(EVALUATE) -a %d,1 $(EMPTY) -n %d -p %d $(GENO) -v 0.2 $(PVE) $(METHOD) $(MODEL) -s %d " sink
                    printf command, m, n, p, seed, n, p, seed
		}
            }
        }
    }
}
endef
export ONE_COMPONENT

joblist:
	echo "$$ONE_COMPONENT"
	awk "$$ONE_COMPONENT" >$@

define TWO_COMPONENT_EQUAL_EFFECT
BEGIN {
    for (logp = 0; logp < 3; logp += 0.5) {
        p1 = 10 ** logp
        p2 = 3 * p1
        if (p2 > 500) {
            continue
        }
        for (seed = 0; seed < 10; seed++) {
            sink = ">$(word 2, $(METHOD))-$(word 2, $(MODEL))-equal-effect-%d-%d-%d.txt\n"
            command = "$(EVALUATE) -a %d,1 -a %d,1 $(DATA) -v 0.2 $(PVE) $(METHOD) $(MODEL) -s %d " sink
            printf command, p1, p2, seed, p1, p2, seed
        }
    }
}
endef
export TWO_COMPONENT_EQUAL_EFFECT

joblist.1:
	awk "$$TWO_COMPONENT_EQUAL_EFFECT" >$@

define TWO_COMPONENT_EQUAL_PROP
BEGIN {
    for (logp = 0; logp < 3; logp++) {
        p = 10 ** logp
        for (seed = 0; seed < 10; seed++) {
            sink = ">$(word 2, $(METHOD))-$(word 2, $(MODEL))-equal-prop-%d-%d.txt\n"
            command = "$(EVALUATE) -a %d,1 -a %d,2 $(DATA) -v 0.2 $(PVE) $(METHOD) $(MODEL) -s %d " sink
            printf command, p, p, seed, p, seed
            printf command, 5 * p, 5 * p, seed, 5 * p, seed
        }
    }
}
endef
export TWO_COMPONENT_EQUAL_PROP

joblist.2:
	awk "$$TWO_COMPONENT_EQUAL_PROP" >$@

joblist.3:
	parallel --dry-run $(EVALUATE) -a 10,1 -n {1} -p 1000 -v 0.2 $(PVE) -K {2} -m logistic $(METHOD) -s {3} ">" $(word 2, $(METHOD))-logistic-ascertained-{1}-1000-{2}-{3}.txt ::: 1000 5000 10000 50000 ::: 0.01 0.05 0.1 0.25 0.5 ::: $$(seq 1 10) >$@

joblist.4: guassian-h2
	parallel --dry-run "$(EVALUATE) -a 10,1 $(DATA) -v {1} --true-pve $(MODEL) $(METHOD) -s {2} >gaussian-h2-1000-1000-{1}-{2}.txt" ::: 0.005 0.01 0.025 0.05 0.075 0.1 0.5 ::: $$(seq 1 10) >$@

gaussian-h2:
	parallel -j1 $(EVALUATE) -a 1,1 -n 1000 -p 100 -v {} --true-pve -M coord -m gaussian -s 1 --write-data {} ::: 0.005 0.01 0.025 0.05

joblist.5:
	parallel --dry-run "$(EVALUATE) -a 10,1 -n 100 -p 100 -v 0.2 --true-pve -M coord -m gaussian --center --rotate -s 0 --$(BOOTSTRAP) {} >coord-gaussian-$(BOOTSTRAP)-{}.txt" ::: $$(seq 0 99) >$@

pre:
	parallel -X ln -sf {} . ::: $(HAPLOTYPES:haplotypes=1kg-phase1)/*.gen.gz $(HAPLOTYPES)/*.sample
	touch $@

chr%:
	parallel --halt now,fail=1 -j1 gtool -S --g $(HAPLOTYPES)/{}_$*.lift.gz --s $(HAPLOTYPES)/{}.sample --og {}_$*.gen.gz --os {}_$*.sample --sample_excl $(HAPLOTYPES)/{}.exclude --exclusion $(HAPLOTYPES)/snps.exclude ::: 58C NBS BD CAD CD HT RA T1D T2D
	gunzip *.sample.gz
	touch $@

imputed-posterior: pre
	parallel -j1 $(EVALUATE) -a 1,1 -n 5000 -p 100 --load-oxstats {58C,NBS,BD,CAD,CD,HT,RA,T1D,T2D}{.sample,_22.gen.gz} --resample -v {} --true-pve -M coord -m gaussian --center -s 0 --write-data {}-5000 ::: 0.005 0.01 0.025 0.05
	parallel -j1 $(EVALUATE) -a 1,1 -n {} -p 100 --load-oxstats {58C,NBS,BD,CAD,CD,HT,RA,T1D,T2D}{.sample,_22.gen.gz} --resample -v 0.01 --true-pve -M coord -m gaussian --center -s 0 --write-data 0.01-{} ::: 5000 10000 50000 100000

array-posterior: chr22
	parallel -j1 $(EVALUATE) -a 1,1 -n 5000 -p 100 --load-oxstats {58C,NBS,BD,CAD,CD,HT,RA,T1D,T2D}_22.{sample,gen.gz} --resample -v {} --true-pve -M coord -m gaussian --center -s 0 --write-data {}-5000 ::: 0.005 0.01 0.025 0.05
	parallel -j1 $(EVALUATE) -a 1,1 -n {} -p 100 --load-oxstats {58C,NBS,BD,CAD,CD,HT,RA,T1D,T2D}_22.{sample,gen.gz} --resample -v 0.01 --true-pve -M coord -m gaussian --center -s 0 --write-data 0.01-{} ::: 5000 10000 50000 100000

annotations.txt:
	zcat 58C_01.gen.gz | awk -vOFS='\t' '{print "chr"$$1, $$3, $$3 + length($$5) - length($$4)}' | bedtools intersect -sorted -nobuf -c -a stdin -b /broad/compbio/aksarkar/projects/roadmap/data/roadmap/core-features/Enh/E042.bed.gz | cut -f4 >$@

clean:
	rm -f *.txt *.txt.gz

.DELETE_ON_ERROR:

.PHONY: gaussian-h2 IL27
