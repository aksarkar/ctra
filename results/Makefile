MODEL := gaussian --center
METHOD := coord
PVE := --true-pve
GENO :=

%.txt.gz:
	awk 'FNR == 1 {sub("./$*-", "", FILENAME); sub(".txt", "", FILENAME); gsub("-", " ", FILENAME)} {print FILENAME, FNR, $$1}' ./$*-*.txt | sort -k1,4n | gzip >$@

define ONE_COMPONENT
BEGIN {
    for (logp=3; logp < 5; logp += 1) {
        for (np=.25; np <= 2; np += 0.25) {
            for (seed=0; seed < 10; seed += 1) {
                m=int(10 ^ (logp - 2));
                p = 10^logp; n=np * p;
		if ("$(GENO)" != "" && n < 16120) {
                    printf "ctra-evaluate -a %d,1 -n %d -p %d $(GENO) -v 0.2 $(PVE) -M $(METHOD) -m $(MODEL) -s %d >$(word 1, $(MODEL))-sample-size-%d-%d-%d.txt\n", m, n, p, seed, n, p, seed
		}
            }
        }
    }
}
endef
export ONE_COMPONENT

joblist.5:
	parallel --dry-run "ctra-evaluate -a 10,1 -n 1000 -p 1000 -v {1} --true-pve -m $(MODEL) -M $(METHOD) -s {2} >gaussian-h2-1000-1000-{1}-{2}.txt" ::: 0.005 0.01 0.025 0.05 0.075 0.1 0.5 ::: $$(seq 1 10) >$@

joblist.4:
	parallel --dry-run ctra-evaluate -a 10,1 -n {1} -p 1000 -v 0.2 $(PVE) -K {2} -m logistic -M $(METHOD) -s {3} ">" logistic-ascertained-{1}-1000-{2}-{3}.txt ::: 1000 5000 10000 50000 ::: 0.01 0.05 0.1 0.25 0.5 ::: $$(seq 1 10) >$@

joblist.3:
	awk "$$ONE_COMPONENT" >$@

define TWO_COMPONENT_EQUAL_PROP
BEGIN {
    for (logp = 0; logp < 3; logp++) {
        p = 10 ** logp
        for (seed = 0; seed < 10; seed++) {
            sink = ">$(word 1, $(METHOD))-$(word 1, $(MODEL))-equal-prop-%d-%d.txt\n"
            command = "ctra-evaluate -a %d,1 -a %d,2 -n 1000 -p 1000 -v 0.2 $(PVE) -M $(METHOD) -m $(MODEL) -s %d " sink
            printf command, p, p, seed, p, seed
            printf command, 5 * p, 5 * p, seed, 5 * p, seed
        }
    }
}
endef
export TWO_COMPONENT_EQUAL_PROP

joblist.1:
	awk "$$TWO_COMPONENT_EQUAL_PROP" >$@

define TWO_COMPONENT_EQUAL_EFFECT
BEGIN {
    for (logp = 0; logp < 3; logp += 0.5) {
        p1 = 10 ** logp
        p2 = 3 * p1
        if (p2 > 500) {
            continue
        }
        for (seed = 0; seed < 10; seed++) {
            sink = ">$(word 1, $(METHOD))-$(word 1, $(MODEL))-equal-effect-%d-%d-%d.txt\n"
            command = "ctra-evaluate -a %d,1 -a %d,1 -n 1000 -p 1000 -v 0.2 $(PVE) -M $(METHOD) -m $(MODEL) -s %d " sink
            printf command, p1, p2, seed, p1, p2, seed
        }
    }
}
endef
export TWO_COMPONENT_EQUAL_EFFECT

joblist:
	awk "$$TWO_COMPONENT_EQUAL_EFFECT" >$@

clean:
	rm -f *.txt *.txt.gz

gaussian-h2:
	parallel -j1 ctra-evaluate -a 1,1 -n 1000 -p 100 -v {} --true-pve -M coord -m gaussian -s 1 --write-data {} ::: 0.005 0.01 0.025 0.05

IL27:
	parallel --halt now,fail=1 -j1 gtool -S --g {}_16_chiamo.gz --s {}.sample --og {}_IL27.gen.gz --os {}_IL27.sample --start 18510683 --end 38518155 --sample_excl {}.exclude --exclusion snps.exclude ::: 58C NBS BD CAD CD HT RA T1D T2D

.PHONY: gaussian-h2 IL27
