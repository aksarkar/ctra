MODEL := guassian
METHOD := coord

%.txt.gz:
	awk 'FNR == 1 {split(FILENAME, a, "-"); sub(".txt", "", a[6])} {print a[3], a[4], a[5], a[6], FNR, $$1}' ./$*-*.txt | sort -k1,4n | gzip >$@

define T
BEGIN {
    for (logp=3; logp < 6; logp += 1) {
        for (logn=2; logn < logp; logn += 0.5) {
            for (seed=0; seed < 10; seed += 1) {
                m=int(10 ^ (logp - 2));
                p = 10^logp; n=10^logn;
                printf "ctra-evaluate -a %d,1 -n %d -p %d -v 0.2 --true-pve -M $(METHOD) -m $(MODEL) -s %d >$(word 1, $(MODEL))-sample-size-%d-%d-%d.txt\n", m, n, p, seed, n, p, seed
            }
        }
    }
}
endef
export T

joblist.3:
	awk "$$T" >$@

joblist.2:
	parallel --dry-run ctra-evaluate -a 100,1 -n {1} -p 1000 -v {2} --true-pve -m $(MODEL) -M $(METHOD) -s {3} -l INFO ">" $(word 1, $(MODEL))-one-component-100-{1}-{2}-{3}.txt ::: 250 500 1000 1500 2000 ::: 0.2 0.4 0.6 ::: $$(seq 1 10) >$@

joblist.1: 
	parallel --dry-run ctra-evaluate -a 100,{1} -a 100,{2} -n 500 -p 1000 -v {3} --true-pve -m $(MODEL) -M $(METHOD) -s {4} -l INFO ">" $(word 1, $(MODEL))-equal-prop-{1}-{2}-{3}-{4}.txt ::: 1 ::: .25 .5 .75 1 ::: 0.2 0.4 0.6 ::: $$(seq 1 10) >$@

joblist:
	parallel --dry-run ctra-evaluate -a {1},1 -a {2},1 -n 500 -p 1000 -v {3} --true-pve -m $(MODEL) -M $(METHOD) -s {4} -l INFO ">" $(MODEL)-equal-effect-{1}-{2}-{3}-{4}.txt ::: 100 200 300 400 500 ::: 0 250 500 ::: 0.2 0.4 0.6 ::: $$(seq 1 10) >$@
