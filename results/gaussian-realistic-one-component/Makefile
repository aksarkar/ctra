gaussian-realistic-one-component: joblist
	test -s joblist
	until pm joblist -l h_vmem=32G -pe smp 2-16 -N gaussian-realistic-one-component; do \
		bash $(HOME)/code/util/redo.sh gaussian-realistic-one-component.*.joblog >redo; \
		rm -f *.joblog; \
		mv redo joblist; \
		done;
	touch $@

joblist: chr22.hdf5
	parallel --dry-run $(EVALUATE) --write-result {#}.pkl -a {1},1 --permute-causal -H $< -n 14162 -p 129877 -v 0.5 -m gaussian -s {2} ::: 100 1000 10000 ::: $$(seq 1 $(TRIALS)) >$@

chr22.hdf5:
	ln -s ../$@ .

clean:
	rm -f gaussian-realistic-one-component joblist gaussian-realistic-one-component.* *.pkl

.PHONY: clean
