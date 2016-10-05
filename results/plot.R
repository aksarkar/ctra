devtools::load_all('/broad/compbio/aksarkar/projects/frea-R/')

panelheight <- 40

pcgc_example <- function(result_file) {
    enrichment <- read.delim(result_file, header=F)
    p <- (ggplot(enrichment, aes(x=factor(V1), y=V6, ymin=V6-V7, ymax=V6+V7)) +
          labs(x='Genetic architecture', y='Per-SNP heritability enrichment') +
          geom_pointrange(size=I(.25)) +
          geom_hline(yintercept=1, size=I(.25), linetype='dashed') +
          scale_y_continuous(limits=c(0, 2)) +
          theme_nature)
    Cairo(file='pcgc-enrichment-example.pdf', type='pdf', height=panelheight, width=panelheight, units='mm')
    print(p)
    dev.off()
}
pcgc_example('/broad/compbio/aksarkar/projects/ctra/results/pcgc-enrichment-example.txt')

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
sample_size('/broad/compbio/aksarkar/projects/ctra/results/pcgc-gaussian-sample-size.txt.gz')
sample_size('/broad/compbio/aksarkar/projects/ctra/results/matlab-gaussian-sample-size.txt.gz')
sample_size('/broad/compbio/aksarkar/projects/ctra/results/matlab-logistic-sample-size.txt.gz')
sample_size('/broad/compbio/aksarkar/projects/ctra/results/dsvi-logistic-sample-size.txt.gz')

equal_effect <- function(result_file) {
    result <- (read.table(gzfile(result_file), se=' ') %>%
               dplyr::select(p1=V1, p2=V2, seed=V3, comp=V4, prop=V5) %>%
               dplyr::group_by(p1, comp) %>%
               dplyr::summarize(pi_hat=mean(prop), se=sqrt(var(prop))) %>%
               dplyr::filter(pi_hat - se > 0))
    p <- (ggplot(data=result, aes(x=p1 / 500, y=pi_hat, ymin=pi_hat - se,
                                  ymax=pi_hat + se, group=comp,
                                  color=factor(comp))) +
          labs(x=expression(paste('True ', pi)),
               y=expression(paste('Posterior mean ', pi)), color='Annotation') +
          geom_line() +
          geom_linerange(size=.25, position=position_dodge(width=0.02)) +
          geom_abline(data=data.frame(comp=c(1, 2), intercept=c(0, log10(2)), slope=c(1, 1)),
                      aes(slope=slope, intercept=intercept, group=factor(comp), color=factor(comp)),
                      size=I(.1), linetype='dashed') +
          scale_color_brewer(palette='Dark2') +
          scale_x_continuous(breaks=10 ^ seq(-3, 0, 1), trans='log10') +
          scale_y_continuous(breaks=10 ^ seq(-3, 0, 1), trans='log10') +
          theme_nature +
          theme(legend.position=c(.6, 1),
                legend.justification=c(1, 1),
                plot.margin=unit(c(0, 2, 0, 0), 'mm')))
    Cairo(file=sub('.txt.gz', '.pdf', result_file), type='pdf',
          height=panelheight, width=panelheight, units='mm')
    print(p)
    dev.off()
}
equal_effect('/broad/compbio/aksarkar/projects/ctra/results/coord-gaussian-equal-effect.txt.gz')

equal_effect_ratio_prop <- function(result_file) {
    result <- (read.table(gzfile(result_file), sep=' ') %>%
               dplyr::select(p=V1, seed=V3, comp=V4, prop=V5) %>%
               dplyr::group_by(p, seed) %>%
               dplyr::do(data.frame(p=.$p, ratio=.$prop[2]/.$prop[1])))
    p <- (ggplot(data=result, aes(x=p / 500, y=ratio, group=p)) +
          labs(x=expression(pi[1]),
               y=expression(pi[2] / pi[1])) +
          geom_boxplot(size=I(.25), outlier.size=.25) +
          geom_hline(yintercept=2, size=.1, linetype='dashed') +
          scale_x_continuous(breaks=10 ^ seq(-3, 0, 1), trans='log10') +
          theme_nature +
          theme(legend.position=c(.6, 1),
                legend.justification=c(1, 1),
                plot.margin=unit(c(0, 2, 0, 0), 'mm')))
    Cairo(file=sub('.txt.gz', '-ratio-prop.pdf', result_file), type='pdf',
          height=panelheight, width=panelheight, units='mm')
    print(p)
    dev.off()
}
equal_effect_ratio_prop('/broad/compbio/aksarkar/projects/ctra/results/coord-gaussian-equal-effect.txt.gz')

equal_prop <- function(result_file) {
    result <- (read.table(gzfile(result_file), sep=' ') %>%
                   dplyr::select(p=V1, seed=V2, comp=V3, prop=V4) %>%
                   dplyr::group_by(p, comp) %>%
                   dplyr::summarize(pi_hat=mean(prop), se=sqrt(var(prop))))
    p <- (ggplot(result, aes(x=p/500, y=pi_hat, ymin=pi_hat - se, ymax=pi_hat + se,
                             group=comp, color=factor(comp))) +
          labs(x=expression(paste('True ', pi)),
               y=expression(paste('Posterior mean ', pi)), color='Annotation') +
          geom_line() +
          geom_linerange(size=.25, position=position_dodge(width=0.02)) +
          geom_abline(intercept=0, slope=1, color='black', size=I(.1), linetype='dashed') +
          scale_color_brewer(palette='Dark2') +
          scale_x_continuous(breaks=10 ^ seq(-3, 0, 1), trans='log10') +
          scale_y_continuous(breaks=10 ^ seq(-3, 0, 1), trans='log10') +
          theme_nature +
          theme(legend.position=c(.6, 1),
                legend.justification=c(1, 1),
                plot.margin=unit(c(0, 2, 0, 0), 'mm')))
    Cairo(file=sub('.txt.gz', '.pdf', result_file), type='pdf',
          height=panelheight, width=panelheight, units='mm')
    print(p)
    dev.off()
}
equal_prop('/broad/compbio/aksarkar/projects/ctra/results/coord-gaussian-equal-prop.txt.gz')

ascertainment <- function(result_file) {
    pihat <- (read.table(gzfile(result_file), sep=' ') %>%
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
ascertainment('/broad/compbio/aksarkar/projects/ctra/results/logistic-ascertained.txt.gz')

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

bvs_posterior <- function(x, y, t0, t1, sigma, rho, scale) {
    (sum(dnorm(y, x[,1:2] %*% c(t0, t1), sigma, log=TRUE)) +
     dnorm(t0, 0, scale, log=TRUE) +
     dnorm(t1, 0, scale, log=TRUE))
}

posterior_contour <- function(args) {
    dir_ = args$dir_[1]
    pi_ = args$pi_[1]
    X <- as.matrix(read.table(paste(dir_, '/genotypes.txt', sep=''),
                              header=F, sep=' '))
    Y <- as.matrix(read.table(paste(dir_, '/phenotypes.txt', sep=''),
                              header=F))
    true_theta <- as.matrix(read.table(paste(dir_, '/theta.txt', sep='')))
    theta_ml <- data.frame(as.list(glm(Y ~ X)$coefficients))
    vx <- var(X %*% true_theta)
    sigma <- sqrt(var(Y) - vx)
    pve <- vx / var(Y)
    tau <- (1 - pve) * pi_ * sum(apply(X, 2, var)) / pve
    scale <- sqrt(1 / tau)
    rho <- 1
    theta0 <- seq(-1, 1, length.out=100)
    posterior <- (
        expand.grid(theta0, theta0) %>%
        dplyr::select(x=Var1, y=Var2) %>%
        dplyr::group_by(x, y) %>%
        dplyr::mutate(z=bvs_posterior(X, Y, x, y, sigma, pi_, scale))
    )
    p <- (ggplot(posterior, aes(x=x, y=y)) +
          geom_contour(aes(z=z, color=..level..), bins=20, size=I(.1)) +
          geom_point(data=data.frame(x=true_theta[1, 1],
                                     y=true_theta[2, 1]),
                     size=I(.5), shape=I(4)) +
          geom_point(data=theta_ml, aes(x=XV1, y=XV2), color='red',
                     size=I(.5), shape=I(4)) +
          geom_point(data=posterior[which.max(posterior$z),], color='blue',
                     size=I(.5), shape=I(4)) +
          geom_hline(yintercept=0, size=I(.1)) +
          geom_vline(xintercept=0, size=I(.1)) +
          scale_color_gradient(low='#fee8c8', high='#e34a33') +
          labs(title=substitute(paste(h^2, '=', pve), list(pve=sprintf('%.3f', pve))),
               x=expression(theta[0]), y=expression(theta[1])) +
          scale_x_continuous(expand=c(0, 0)) +
          scale_y_continuous(expand=c(0, 0)) +
          theme_nature +
          theme(plot.title=element_text(),
                plot.margin=unit(c(0, 2, 0, 0), 'mm')))
    Cairo(file=paste(dir_, 'llik.pdf', sep='/'), type='pdf', width=panelheight, height=panelheight, units='mm')
    print(p)
    dev.off()
}

plot_posterior_contour <- function(root) {
    pve <- c('0.005', '0.01', '0.025', '0.05')
    pi_ <- 0.01
    plots <- (expand.grid(pve, pi_) %>%
              dplyr::mutate(dir_=paste(root, pve, sep='/'), pi_=Var2) %>%
              dplyr::group_by(dir_, pi_) %>%
              dplyr::do(plot=posterior_contour(.)))
}
plot_posterior_contour('/broad/hptmp/aksarkar/test')

ep_posterior <- function(root) {
    x <- as.matrix(read.table(paste(root, 'genotypes.txt', sep='/')))
    y <- as.matrix(read.table(paste(root, 'phenotypes.txt', sep='/')))
    vx <- sum(apply(x, 2, var))
    p0 <- logistic(seq(-3, 0, .25) * log(10))
    models <- (data.frame(p0=p0) %>%
               dplyr::mutate(beta=1 / var(y), v=.2 / (1 - .2) / (p0 * vx)) %>%
               dplyr::rowwise() %>%
               dplyr::do(logw=do.call(epBVSinternal, c(list(x, y), .))$evidence))
    logw <- unlist(models$logw)
    logw <- logw - max(logw)
    logw <- logw - log(sum(exp(logw)))
    posterior <- (ggplot(data.frame(p0=p0, logw=logw), aes(x=p0, y=logw)) +
                  geom_line() +
                  labs(x=expression(pi), y=expression(ln(p(pi * "|" * x)))) +
                  theme_nature)
    Cairo(file=paste(root, 'ep-posterior.pdf', sep='/'), type='pdf',
          width=panelheight, height=panelheight, units='mm')
    print(posterior)
    dev.off()
}
ep_posterior('/broad/hptmp/aksarkar/test')
