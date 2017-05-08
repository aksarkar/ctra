all: joblist
	until pm joblist -l h_vmem=4G -q long -N gaussian-idealized-one-component; do \
		bash $(HOME)/code/util/redo.sh gaussian-idealized-one-component.*.joblog >redo; \
		rm *.joblog; \
		mv redo joblist; \
		done;

joblist:
	parallel --dry-run $(EVALUATE) --write-result {#}.pkl -a {1},1 -n 5000 -p 10000 -v 0.5 -m gaussian -s {2} ::: 100 1000 ::: $$(seq 1 $(TRIALS)) >$@

clean:
	rm gaussian-idealized-one-component.*
	rm *.pkl
