all: joblist
	until pm joblist -l h_vmem=4G -q long -N logistic-idealized-one-component; do \
		bash $(HOME)/code/util/redo.sh logistic-idealized-one-component.*.joblog >redo || true; \
		rm -f *.joblog; \
		mv redo joblist; \
		done;
	touch $@

joblist:
	parallel --dry-run $(EVALUATE) --write-result {#}.pkl -a {1},1 -n 7500 -p 10000 -v 0.5 -m logistic -K 0.01 -s {2} ::: 10 100 1000 ::: $$(seq 1 $(TRIALS)) >$@

clean:
	rm -f all joblist logistic-idealized-one-component.* *.pkl
