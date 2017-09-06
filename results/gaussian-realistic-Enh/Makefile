TRIALS := 10

gaussian-realistic-Enh: joblist
	test -s joblist
	until pm joblist -R y -l h_vmem=32G -pe smp 4-16 -N gaussian-realistic-Enh; do \
		bash $(HOME)/code/util/redo.sh gaussian-realistic-Enh.*.joblog >redo; \
		rm -f *.joblog; \
		mv redo joblist; \
		done;
	touch $@

joblist: chr22.hdf5 Enh.txt.gz
	parallel --dry-run $(EVALUATE) --write-result {#}.pkl --annotation-matrix $(word 2, $^) --annotation-matrix-column {3} -a {1},1 -a 0,1 --permute-causal -H $< -n 14162 -p 129877 -v 0.5 -m gaussian -s {2} ::: 100 1000 ::: $$(seq 1 $(TRIALS)) ::: 87 78 88 21 >$@

Enh.txt.gz chr22.hdf5:
	ln -s ../$@ .

clean:
	rm -f gaussian-realistic-Enh joblist gaussian-realistic-Enh.* *.pkl

.PHONY: clean
