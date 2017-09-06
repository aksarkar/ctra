define TWO_COMPONENT
BEGIN {
    command = "$(EVALUATE) --write-result %d.pkl --annotation-vector $(word 2, $^) -a %d,%d -a %d,%d --permute-causal -H $< -n 14162 -p 129877 -v 0.5 -m gaussian -s %d\n"
    sink = 0
    for (logp = 1; logp < 3; logp++) {
        for (seed = 0; seed < 50; seed++) {
            p = 10 ** logp
            printf command, sink, .33 * p, 1, .67 * p, 1, seed
            sink += 1
            printf command, sink, .5 * p, 1, .5 * p, 2, seed
        }
    }
}
endef
export TWO_COMPONENT

gaussian-realistic-two-component: joblist
	test -s joblist
	until pm joblist -l h_vmem=32G,d_rt=3600 -pe smp 4-16 -N gaussian-realistic-two-component; do \
		bash $(HOME)/code/util/redo.sh gaussian-realistic-two-component.*.joblog >redo; \
		rm -f *.joblog; \
		mv redo joblist; \
		done;
	touch $@

joblist: chr22.hdf5 77.txt
	awk "$$TWO_COMPONENT" >$@

77.txt chr22.hdf5:
	ln -s ../$@ .

clean:
	rm -f gaussian-realistic-two-component joblist gaussian-realistic-two-component.* *.pkl

.PHONY: clean
