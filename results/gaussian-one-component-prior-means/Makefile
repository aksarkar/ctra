gaussian-one-component-prior-means: joblist
	until pm joblist -l h_vmem=4G -N gaussian-one-component-prior-means; do \
		bash $(HOME)/code/util/redo.sh gaussian-one-component-prior-means.*.joblog >redo; \
		rm -f *.joblog; \
		mv redo joblist; \
		done;
	touch $@

joblist:
	parallel --dry-run $(EVALUATE) --write-result {#}.pkl -a 1000,1 -n 7500 -p 10000 --prior-mean-b {1} --prior-mean-c {2} -v 0.5 -m gaussian -s {3} ::: $$(seq -5 0) ::: 0 5 10 50 100 ::: $$(seq 1 $(TRIALS)) >$@

clean:
	rm -f gaussian-one-component-prior-means joblist gaussian-one-component-prior-means.* *.pkl

.PHONY: clean
