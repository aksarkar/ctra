gaussian-one-component-versus-sample-size: joblist
	until pm joblist -l h_vmem=64G -N gaussian-one-component-versus-sample-size; do \
		bash $(HOME)/code/util/redo.sh gaussian-one-component-versus-sample-size.*.joblog >redo; \
		rm -f *.joblog; \
		mv redo joblist; \
		done;
	touch $@

joblist:
	parallel --dry-run $(EVALUATE) --write-result {#}.pkl -a 1000,1 -n {1} -p 10000 -v 0.5 -m gaussian -s {2} ::: 1000 5000 10000 50000 100000 ::: $$(seq 1 $(TRIALS)) >$@

clean:
	rm -f gaussian-one-component-versus-sample-size joblist gaussian-one-component-versus-sample-size.* *.pkl

.PHONY: clean
