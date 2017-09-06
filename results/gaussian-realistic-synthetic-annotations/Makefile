gaussian-realistic-synthetic-annotations: joblist
	test -s joblist
	until pm joblist -l h_vmem=32G -pe smp 4-12 -N gaussian-realistic-synthetic-annotations; do \
		bash $(HOME)/code/util/redo.sh gaussian-realistic-synthetic-annotations.*.joblog >redo; \
		rm -f *.joblog; \
		mv redo joblist; \
		done;
	touch $@

joblist: chr22.hdf5 synthetic-annotation-1kb.txt synthetic-annotation-1mb.txt
	parallel --dry-run $(EVALUATE) --write-result {#}.pkl -A {1} -a {2},1 -a 0,1 --permute-causal -H $< -n 14162 -p 129877 -v 0.5 -m gaussian -s {3} ::: synthetic-annotation-1kb.txt synthetic-annotation-1mb.txt ::: 100 1000 10000 ::: $$(seq 1 $(TRIALS))  >$@

chr22.hdf5 synthetic-annotation-1kb.txt synthetic-annotation-1mb.txt:
	ln -s ../$@ .

clean:
	rm -f gaussian-realistic-synthetic-annotations joblist gaussian-realistic-synthetic-annotations.* *.pkl

.PHONY: clean
