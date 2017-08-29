gaussian-idealized-one-component: joblist
	until pm joblist -l h_vmem=4G -q long -N gaussian-idealized-one-component; do \
		bash $(HOME)/code/util/redo.sh gaussian-idealized-one-component.*.joblog >redo; \
		rm -f *.joblog; \
		mv redo joblist; \
		done;
	touch $@

joblist:
	parallel --dry-run $(EVALUATE) --write-result {#}.pkl -a {1},1 -n 7500 -p 10000 -v 0.5 -m gaussian -s {2} ::: 10 100 1000 ::: $$(seq 1 $(TRIALS)) >$@

clean:
	rm -f gaussian-idealized-one-component joblist gaussian-idealized-one-component.* *.pkl

.PHONY: clean
