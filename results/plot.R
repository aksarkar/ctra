library(frea)
library(gridExtra)

panelheight <- 40

pcgc_example <- function() {
    enrichment <- read.delim('/broad/compbio/aksarkar/projects/ctra/results/pcgc-enrichment-example.txt', header=F)
    p <- (ggplot(enrichment, aes(x=factor(V1), y=V6, ymin=V6-V7, ymax=V6+V7)) +
          labs(x='Genetic architecture', y='Per-SNP heritability enrichment') +
          geom_pointrange(size=I(.25)) +
          geom_hline(yintercept=1, size=I(.25), linetype='dashed') +
          scale_y_continuous(limits=c(0, 2)) +
          theme_nature)
    Cairo(file='pcgc-enrichment-example.pdf', type='pdf', height=panelheight, width=panelheight, units='mm')
    print(p)
    dev.off()
}()

sample_size <- function(result_file) {
    result <- (read.table(gzfile(result_file), sep=' ') %>%
               dplyr::select(n=V1, p=V2, seed=V3, pi_=V5) %>%
               dplyr::group_by(n, p) %>%
               dplyr::summarize(se=sqrt(var(pi_)), pi_=mean(pi_)))
    my_plot <- (ggplot(result, aes(x=n/p, y=pi_, ymin=pi_-se, ymax=pi_+se, color=factor(p))) +
                labs(x='Samples n / Variants p', y=expression(paste('Posterior mean ', pi)), color='p') +
                scale_color_brewer(palette='Dark2') +
                geom_line(size=.25) +
                geom_linerange(size=.25, position=position_dodge(width=0.05)) +
                geom_hline(yintercept=.01, size=I(.25), linetype='dashed') +
                theme_nature +
                theme(legend.position=c(1, 1),
                      legend.justification=c(1, 1)))
    Cairo(file=sub('.txt.gz', '.pdf', result_file), type='pdf', height=panelheight, width=panelheight, units='mm')
    print(my_plot)
    dev.off()
}
sample_size('/broad/compbio/aksarkar/projects/ctra/results/gaussian-sample-size.txt.gz')

sample_size('/broad/compbio/aksarkar/projects/ctra/results/matlab-gaussian-sample-size.txt.gz')
sample_size('/broad/compbio/aksarkar/projects/ctra/results/matlab-logistic-sample-size.txt.gz')


equal_effect <- function(result_file) {
    equal_effect <- (
        read.table(gzfile(result_file), sep=' ') %>%
        dplyr::select(p1=V1, p2=V2, seed=V4, comp=V5, pi_=V6) %>%
        dplyr::group_by(p1, p2, comp) %>%
        dplyr::summarize(pi_hat=mean(pi_), se=sqrt(var(pi_)))
    )
    p <- (ggplot(equal_effect, aes(x=p1/500, y=pi_hat, ymin=pi_hat - se,
                                   ymax=pi_hat + se, group=comp,
                                   color=factor(comp))) +
          labs(x=expression(paste('True ', pi)),
               y=expression(paste('Posterior mean ', pi)), color='Annotation') +
          geom_line() +
          geom_linerange(size=.25, position=position_dodge(width=0.05)) +
          scale_x_log10(limits=c(1e-3, 1), breaks=c(1e-3, 1e-2, 1e-1, 1)) +
          scale_y_log10(limits=c(1e-4, 1), breaks=c(1e-3, 1e-2, 1e-1, 1)) +
          scale_color_brewer(palette='Dark2') +
          geom_abline(aes(intercept=intercept, slope=slope, color=factor(comp)),
                      data=data.frame(comp=c(1, 2), intercept=c(0, -1),
                                      slope=c(1, 0)),
                      size=I(.1), linetype='dashed') +
          theme_nature +
          theme(legend.position=c(1, .5),
                legend.justification=c(1, 1)))
    Cairo(file=sub('.txt.gz', '.pdf', result_file), type='pdf',
          height=panelheight, width=panelheight, units='mm')
    print(p)
    dev.off()
}
equal_effect('/broad/compbio/aksarkar/projects/ctra/results/gaussian-equal-effect.txt.gz')

equal_prop <- function() {
    equal_prop <- (read.table(gzfile('/broad/compbio/aksarkar/projects/ctra/results/equal-prop.txt.gz'), sep=' ') %>%
                   dplyr::select(tau1=V2, h2=V3, seed=V4, comp=V5, pi_=V6) %>%
                   dplyr::filter(comp == 1 & h2 == 0.2) %>%
                   dplyr::group_by(tau1) %>%
                   dplyr::summarize(pi_hat=mean(pi_), se=sqrt(var(pi_))))
    p <- (ggplot(equal_prop, aes(x=tau1, y=pi_hat, ymin=pi_hat-se, ymax=pi_hat+se)) +
          labs(x=expression(paste('Relative effect size ', tau[1] / tau[0])), y=expression(paste('Posterior mean ', pi))) +
          geom_pointrange(size=.25, fatten=1) +
          geom_hline(yintercept=.1, size=I(.25), linetype='dashed') +
          theme_nature)
    Cairo(file='equal-prop.pdf', type='pdf', height=panelheight, width=panelheight, units='mm')
    print(p)
    dev.off()
}()

ascertainment <- function() {
    pihat <- (read.table(gzfile('/broad/compbio/aksarkar/projects/ctra/results/logistic-ascertained.txt.gz'), sep=' ') %>%
              dplyr::select(n=V1, k=V3, seed=V4, pi_=V6) %>%
              dplyr::group_by(n, k) %>%
              dplyr::summarize(pi_hat=mean(pi_), se=sqrt(var(pi_))))
    p <- (ggplot(pihat, aes(x=k, y=pi_hat, ymin=pi_hat-se, ymax=pi_hat+se, color=factor(n))) +
          labs(x='Population prevalence', y=expression(paste('Posterior mean ', pi)), color='n') +
          geom_line(size=.25) +
          geom_hline(yintercept=.01, size=I(.25), linetype='dashed') +
          scale_color_brewer(palette='Dark2') +
          theme_nature +
          theme(legend.position=c(1, 1),
                legend.justification=c(1, 1)))
    Cairo(file='logistic-ascertained.pdf', type='pdf', height=panelheight, width=panelheight, units='mm')
    print(p)
    dev.off()
}
ascertainment()

joint_posterior <- function(weights_file) {
    weights <- read.table(weights_file, header=F, sep=' ')
    p0 <- (ggplot(weights, aes(x=V1, y=log10(V3))) +
          geom_line() +
          labs(x=expression(pi[0]), y=expression(paste(log[10], p, "(", pi[0], "|", x, ",", pi[1], ")"))) +
          facet_wrap(~ V2, nrow=1) +
          theme_nature +
          theme(panel.margin=unit(3, 'mm')))
    p1 <- (ggplot(weights, aes(x=V2, y=log10(V3))) +
          geom_line() +
          labs(x=expression(pi[1]), y=expression(paste(log[10], p, "(", pi[1], "|", x, ",", pi[0], ")"))) +
          facet_wrap(~ V1, nrow=1) +
          theme_nature +
          theme(panel.margin=unit(3, 'mm')))
    Cairo(file=sub('.txt', '.pdf', weights_file), type='pdf', height=2 * panelheight, width=7 * panelheight, units='mm')
    grid.arrange(p0, p1, nrow=2)
    dev.off()
}
joint_posterior('/broad/hptmp/aksarkar/test/weights.txt')

pi_versus_h2 <- function(result_file) {
    result <- (
        read.table(gzfile(result_file), header=F, sep=' ') %>%
        dplyr::select(pve=V3, seed=V4, pi_=V6) %>%
        dplyr::group_by(pve) %>%
        dplyr::summarize(pi_hat=mean(pi_), se=sqrt(var(pi_)))
    )
    p <- (ggplot(result, aes(x=pve, y=pi_hat, ymin=pi_hat - se,
                             ymax=pi_hat + se)) +
          labs(x='Heritability', y=expression(paste('Posterior mean ', pi))) +
          geom_line(size=.25) +
          geom_linerange(size=.25) +
          geom_hline(yintercept=0.01, size=I(.1), linetype='dashed') +
          theme_nature)
    Cairo(file=sub('.txt.gz', '.pdf', result_file), type='pdf', height=panelheight, width=panelheight, units='mm')
    print(p)
    dev.off()
}
pi_versus_h2('/broad/compbio/aksarkar/projects/ctra/results/mcmc-gaussian-h2.txt.gz')
pi_versus_h2('/broad/compbio/aksarkar/projects/ctra/results/gaussian-h2.txt.gz')
pi_versus_h2('/broad/compbio/aksarkar/projects/ctra/results/corrected-tau-gaussian-h2.txt.gz')
pi_versus_h2('/broad/compbio/aksarkar/projects/ctra/results/normalized-gaussian-h2.txt.gz')
pi_versus_h2('/broad/compbio/aksarkar/projects/ctra/results/matlab-gaussian-h2.txt.gz')


pi_prior <- function() {
    x <- seq(0, 1, length.out=100)
    params <- data.frame(expand.grid(pve=c(.01, .05, .1),
                                     nk=c(1e-3, .01, .1)))
    data <- (params %>%
             dplyr::mutate(b=exp(nk / pve) * (1 - pve)) %>%
             dplyr::group_by(pve, nk) %>%
             do({data.frame(x=x, y=dbeta(x, 1, .$b))}))
    p <- (ggplot(as.data.frame(data), aes(x=x, y=y, color=factor(nk))) +
          labs(x=expression(pi), y=expression(f(pi)), color='Size') +
          geom_line() +
          facet_wrap(~pve, scales='free', nrow=1) +
          theme_nature +
          theme(legend.position='right',
                panel.margin=unit(4, 'mm')))
    Cairo(file='test.pdf', type='pdf', width=120, height=40, units='mm')
    print(p)
    dev.off()
}
pi_prior()

gaussian_llik <- function(x, y, t0, t1, sigma, tau) {
    (sum(dnorm(y, x %*% c(t0, t1), sigma, log=TRUE)) +
     dnorm(t0, 0, tau, log=TRUE) +
     dnorm(t1, 0, tau, log=TRUE))
}

likelihood_contour <- function(genotypes, phenotypes, theta) {
    X <- as.matrix(read.table(genotypes, header=F, sep=' '))
    Y <- as.matrix(read.table(phenotypes, header=F))
    true_theta <- as.matrix(read.table('/broad/hptmp/aksarkar/test/theta.txt'))
    vx <- var(X %*% true_theta)
    sigma <- sqrt(var(Y) - vx)
    pve <- vx / var(Y)
    scale <- sqrt(pve / (1 - pve) / sum(apply(X, 2, var)))
    theta0 <- seq(-3, 3, length.out=100)
    likelihood <- (
        expand.grid(theta0, theta0) %>%
        dplyr::select(x=Var1, y=Var2) %>%
        dplyr::group_by(x, y) %>%
        dplyr::mutate(z=gaussian_llik(X, Y, x, y, sigma, scale))
    )
    p <- (ggplot(likelihood, aes(x=x, y=y, z=z)) +
          geom_point(data=data.frame(x=true_theta[1, 1],
                                     y=true_theta[2, 1], z=0)) +
          stat_contour(geom='path', bins=15) +
          labs(title=substitute(paste(h^2, '=', pve, ', ', tau, '=', scale),
                                list(pve=pve, scale=scale)),
               x=expression(theta[0]), y=expression(theta[1])) +
          theme_nature +
          theme(plot.title=element_text()))
    Cairo(file='llik.pdf', type='pdf', width=40, height=40, units='mm')
    print(p)
    dev.off()
}
likelihood_contour('/broad/hptmp/aksarkar/test/0.01/genotypes.txt',
                   '/broad/hptmp/aksarkar/test/0.01/phenotypes.txt',
                   '/broad/hptmp/aksarkar/test/0.01/theta.txt')
