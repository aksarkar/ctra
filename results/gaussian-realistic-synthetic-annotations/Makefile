all: joblist
	test -s joblist
	until pm joblist -l h_vmem=32G -pe smp 2-16 -N gaussian-realistic-synthetic-annotations; do \
		bash $(HOME)/code/util/redo.sh gaussian-realistic-synthetic-annotations.*.joblog >redo; \
		rm -f *.joblog; \
		mv redo joblist; \
		done;
	touch $@

joblist: chr22.hdf5 annotations.txt
	parallel --dry-run $(EVALUATE) --write-result {#}.pkl -A annotations.txt -a {1},1 -a 0,1 --permute-causal -H $< -n 14162 -p 129877 -v 0.5 -m gaussian -s {2} ::: 100 1000 10000 ::: $$(seq 1 $(TRIALS)) >$@

annotations.txt chr22.hdf5:
	ln -s ../$@ .

clean:
	rm -f all joblist gaussian-realistic-synthetic-annotations.* *.pkl

.PHONY: clean
