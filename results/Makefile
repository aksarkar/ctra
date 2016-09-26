MODEL := gaussian
METHOD := coord

%.txt.gz:
	awk 'FNR == 1 {split(FILENAME, a, "-"); sub(".txt", "", a[6])} {print a[3], a[4], a[5], a[6], FNR, $$1}' ./$*-*.txt | sort -k1,4n | gzip >$@

define ONE_COMPONENT
BEGIN {
    for (logp=3; logp < 5; logp += 1) {
        for (np=.25; np <= 2; np += 0.25) {
            for (seed=0; seed < 10; seed += 1) {
                m=int(10 ^ (logp - 2));
                p = 10^logp; n=np * p;
                printf "ctra-evaluate -l DEBUG -a %d,1 -n %d -p %d -v 0.2 --true-pve -M $(METHOD) -m $(MODEL) -s %d >$(word 1, $(MODEL))-sample-size-%d-%d-%d.txt\n", m, n, p, seed, n, p, seed
            }
        }
    }
}
endef
export ONE_COMPONENT

joblist.4:
	parallel --dry-run ctra-evaluate -a 10,1 -n {1} -p 1000 -v 0.2 --true-pve -K {2} -m logistic -M $(METHOD) -s {3} ">" logistic-ascertained-{1}-1000-{2}-{3}.txt ::: 1000 5000 10000 50000 ::: 0.01 0.05 0.1 0.25 0.5 ::: $$(seq 1 10) >$@

joblist.3:
	awk "$$ONE_COMPONENT" >$@

joblist.2:
	parallel --dry-run ctra-evaluate -a 100,1 -n {1} -p 1000 -v {2} --true-pve -m $(MODEL) -M $(METHOD) -s {3} -l INFO ">" $(word 1, $(MODEL))-one-component-100-{1}-{2}-{3}.txt ::: 250 500 1000 1500 2000 ::: 0.2 0.4 0.6 ::: $$(seq 1 10) >$@

joblist.1: 
	parallel --dry-run ctra-evaluate -a 100,{1} -a 100,{2} -n 500 -p 1000 -v {3} --true-pve -m $(MODEL) -M $(METHOD) -s {4} -l INFO ">" $(word 1, $(MODEL))-equal-prop-{1}-{2}-{3}-{4}.txt ::: 1 ::: .25 .5 .75 1 ::: 0.2 0.4 0.6 ::: $$(seq 1 10) >$@

joblist:
	parallel --dry-run ctra-evaluate -a {1},1 -a {2},1 -n 500 -p 1000 -v {3} --true-pve -m $(MODEL) -M $(METHOD) -s {4} -l INFO ">" $(MODEL)-equal-effect-{1}-{2}-{3}-{4}.txt ::: 100 200 300 400 500 ::: 0 250 500 ::: 0.2 0.4 0.6 ::: $$(seq 1 10) >$@
