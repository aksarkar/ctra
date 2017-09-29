gaussian-one-component-versus-sample-size: joblist
	until pm joblist -l h_vmem=64G -N gaussian-one-component-versus-sample-size; do \
		bash $(HOME)/code/util/redo.sh gaussian-one-component-versus-sample-size.*.joblog >redo; \
		rm -f *.joblog; \
		mv redo joblist; \
		done;
	touch $@

joblist:
	parallel --dry-run $(EVALUATE) --write-result {#}.pkl -a {1},1 -n {2} -p {1}0 -v 0.5 -m gaussian -s {3} ::: 1000 2000 3000 4000 5000 ::: 5000 10000 50000 100000 ::: $$(seq 1 $(TRIALS)) >$@

clean:
	rm -f gaussian-one-component-versus-sample-size joblist gaussian-one-component-versus-sample-size.* *.pkl

.PHONY: clean
