define TWO_COMPONENT_EQUAL_EFFECT
BEGIN {
    command = "$(EVALUATE) --write-result %d.pkl -a %d,%d -a %d,%d -n 7500 -p 10000 -v 0.5 -m logistic -K 0.01 -s %d\n"
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
export TWO_COMPONENT_EQUAL_EFFECT

all: joblist
	test -s $<
	until pm $< -l h_vmem=4G -q long -N logistic-idealized-two-component; do \
		bash $(HOME)/code/util/redo.sh logistic-idealized-two-component.*.joblog >redo; \
		rm *.joblog; \
		mv redo joblist; \
		done;
	touch $@

joblist:
	awk "$$TWO_COMPONENT_EQUAL_EFFECT" >$@

clean:
	rm -f joblist
	rm -f logistic-idealized-two-component.*
	rm -f *.pkl
