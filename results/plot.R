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
    Cairo(file='pcgc-enrichment-example.pdf', type='pdf', height=panelheight,
          width=panelheight, units='mm')
    print(p)
    dev.off()
}
pcgc_example('/broad/compbio/aksarkar/projects/ctra/results/pcgc-enrichment-example.txt')

sample_size <- function(result_file, thresh_file=NA) {
    result <- (read.table(gzfile(result_file), sep=' ') %>%
               dplyr::select(n=V1, p=V2, seed=V3, pi_=V5) %>%
               dplyr::mutate(variable='pi'))
    if (!is.na(thresh_file)) {
        thresh <- (read.table(gzfile(thresh_file), sep=' ') %>%
                   dplyr::select(n=V1, p=V2, seed=V3, pi_=V5) %>%
                   dplyr::mutate(variable='PIP'))
        result <- rbind(result, thresh)
    }
    my_plot <- (ggplot(result, aes(x=n, y=pi_, color=variable, group=interaction(factor(n), variable))) +
                labs(x='Number of individuals',
                     y=expression(paste('Posterior mean ', pi)), color='') +
                scale_color_brewer(palette='Dark2', labels=c(expression(pi), expression('PIP' > 0.1))) +
                scale_x_continuous(breaks=seq(2500, 10000, 2500)) +
                geom_boxplot(size=I(.1), outlier.size=.25) +
                geom_hline(yintercept=.01, size=I(.25), linetype='dashed') +
                theme_nature +
                theme(legend.position=c(1.2, .5),
                      legend.justification=c(1, 1),
                      plot.margin=grid::unit(rep(2, 4), 'mm')))
    Cairo(file=sub('.txt.gz', '.pdf', result_file), type='pdf',
          height=panelheight, width=panelheight, units='mm')
    print(my_plot)
    dev.off()
}
sample_size('/broad/compbio/aksarkar/projects/ctra/results/wsabi-gaussian-coord-sample-size-10000.txt.gz', '/broad/compbio/aksarkar/projects/ctra/results/wsabi-gaussian-sample-size-pip-thresh.txt.gz')

equal_effect <- function(result_file) {
    result <- (read.table(gzfile(result_file), se=' ') %>%
               dplyr::select(p1=V1, p2=V2, seed=V3, comp=V4, prop=V5))
    p <- (ggplot(data=result, aes(x=p1 / 500, y=prop, color=factor(comp))) +
          labs(x=expression(paste('True ', pi)),
               y=expression(paste('Posterior mean ', pi)), color='Annotation') +
          geom_boxplot(aes(group=interaction(p1, comp)), width=.25, size=.1, outlier.size=.25) +
          geom_abline(intercept=0, slope=1, aes(color=factor(1)), linetype='dashed') +
          geom_abline(intercept=log10(3), slope=1, aes(color=factor(2)), linetype='dashed') +
          scale_color_brewer(palette='Dark2') + #
          scale_x_continuous(breaks=10 ^ seq(-3, 0, 1), trans='log10') +
          scale_y_continuous(breaks=10 ^ seq(-3, 0, 1), trans='log10',
                             oob=scales::squish) +
          theme_nature +
          theme(legend.position=c(.7, 1.1),
                legend.justification=c(1, 1),
                plot.margin=grid::unit(c(0, 2, 0, 0), 'mm')))
    Cairo(file=sub('.txt.gz', '.pdf', result_file), type='pdf',
          height=panelheight, width=panelheight, units='mm')
    print(p)
    dev.off()
}
equal_effect('/broad/compbio/aksarkar/projects/ctra/results/wsabi-coord-gaussian-equal-effect-no-pool.txt.gz')
equal_effect('/broad/compbio/aksarkar/projects/ctra/results/wsabi-coord-gaussian-equal-effect.txt.gz')
equal_effect('/broad/compbio/aksarkar/projects/ctra/results/realistic-coord-gaussian-equal-effect.txt.gz')

equal_effect_ratio_prop <- function(result_file) {
    result <- (read.table(gzfile(result_file), sep=' ') %>%
               dplyr::select(p=V1, seed=V3, comp=V4, prop=V5) %>%
               dplyr::group_by(p, seed) %>%
               dplyr::do(data.frame(p=.$p, ratio=.$prop[2]/.$prop[1])))
    p <- (ggplot(data=result, aes(x=p / 500, y=ratio, group=p)) +
          labs(x=expression(pi[1]),
               y=expression(pi[2] / pi[1])) +
          geom_boxplot(size=I(.25), outlier.size=.25) +
          geom_hline(yintercept=3, size=.1, linetype='dashed') +
          scale_x_continuous(breaks=10 ^ seq(-3, 0, 1), trans='log10') +
          theme_nature +
          theme(legend.position=c(.6, 1),
                legend.justification=c(1, 1),
                plot.margin=grid::unit(c(0, 2, 0, 0), 'mm')))
    Cairo(file=sub('.txt.gz', '-ratio-prop.pdf', result_file), type='pdf',
          height=panelheight, width=panelheight, units='mm')
    print(p)
    dev.off()
}
equal_effect_ratio_prop('/broad/compbio/aksarkar/projects/ctra/results/wsabi-coord-gaussian-equal-effect.txt.gz')
equal_effect_ratio_prop('/broad/compbio/aksarkar/projects/ctra/results/wsabi-coord-gaussian-equal-effect-no-pool.txt.gz')

equal_prop <- function(result_file) {
    result <- (read.table(gzfile(result_file), sep=' ') %>%
                   dplyr::select(p=V1, seed=V2, comp=V3, prop=V4))
    p <- (ggplot(result, aes(x=p/500, y=prop, color=factor(comp))) +
          labs(x=expression(paste('True ', pi)),
               y=expression(paste('Posterior mean ', pi)), color='Annotation') +
          geom_boxplot(aes(group=interaction(p, comp)), width=.25, size=.1, outlier.size=.25) +
          geom_abline(intercept=0, slope=1, color='black', size=I(.1),
                      linetype='dashed') +
          scale_color_brewer(palette='Dark2') +
          scale_x_continuous(breaks=10 ^ seq(-3, 0, 1), trans='log10') +
          scale_y_continuous(breaks=10 ^ seq(-3, 0, 1), trans='log10') +
          theme_nature +
          theme(legend.position=c(.65, 1.1),
                legend.justification=c(1, 1),
                plot.margin=grid::unit(c(0, 2, 0, 0), 'mm')))
    Cairo(file=sub('.txt.gz', '.pdf', result_file), type='pdf',
          height=panelheight, width=panelheight, units='mm')
    print(p)
    dev.off()
}
equal_prop('/broad/compbio/aksarkar/projects/ctra/results/wsabi-coord-gaussian-equal-prop-no-pool.txt.gz')
equal_prop('/broad/compbio/aksarkar/projects/ctra/results/wsabi-coord-gaussian-equal-prop.txt.gz')
equal_prop('/broad/compbio/aksarkar/projects/ctra/results/realistic-coord-gaussian-equal-prop.txt.gz')

equal_prop_propose_tau <- function(result_file) {
    result <- (read.table(gzfile(result_file), sep=' ') %>%
               dplyr::select(p=V1, seed=V2, comp=V3, prop=V4) %>%
               dplyr::filter(comp == 1))
    p <- (ggplot(result, aes(x=p/1000, y=prop, color=factor(comp))) +
          labs(x=expression(paste('True ', pi)),
               y=expression(paste('Posterior mean ', pi)), color='Annotation') +
          geom_boxplot(aes(group=interaction(p, comp)), width=.25, size=.1, outlier.size=.25) +
          geom_abline(intercept=0, slope=1, color='black', size=I(.1),
                      linetype='dashed') +
          scale_color_brewer(palette='Dark2') +
          scale_x_continuous(breaks=10 ^ seq(-3, 0, 1), trans='log10') +
          scale_y_continuous(breaks=10 ^ seq(-3, 0, 1), trans='log10') +
          theme_nature +
          theme(legend.position=c(.6, 1),
                legend.justification=c(1, 1),
                plot.margin=grid::unit(c(0, 2, 0, 0), 'mm')))
    Cairo(file=sub('.txt.gz', '.pdf', result_file), type='pdf',
          height=panelheight, width=panelheight, units='mm')
    print(p)
    dev.off()
}
equal_prop_propose_tau('/broad/compbio/aksarkar/projects/ctra/results/wsabi-coord-gaussian-equal-prop-propose-tau.txt.gz')

ascertainment <- function(varbvs_file, dsvi_file) {
    varbvs <- (read.table(gzfile(varbvs_file), sep=' ') %>%
               dplyr::select(n=V1, k=V3, seed=V4, varbvs=V6))
    dsvi <- (read.table(gzfile(dsvi_file), sep=' ') %>%
             dplyr::select(n=V1, k=V3, seed=V4, dsvi=V6) %>%
             dplyr::inner_join(varbvs, by=c('n', 'k', 'seed')))
    p <- (ggplot(dsvi, aes(x=seed, y=dsvi)) +
          labs(x='Trial',
               y=expression(paste('Posterior mean ', pi)), color='n') +
          geom_point(size=I(.5)) +
          geom_linerange(aes(ymax=pmax(dsvi, varbvs), ymin=pmin(dsvi, varbvs)), size=.25) +
          geom_hline(yintercept=.01, size=I(.25), linetype='dashed') +
          scale_x_continuous(breaks=seq(1, 10)) +
          scale_color_brewer(palette='Dark2') +
          facet_wrap(n ~ k, scales='free', nrow=4) +
          theme_nature +
          theme(panel.margin=grid::unit(2, 'mm'),
                panel.background=element_rect(colour='black'))
    )
    Cairo(file=sub('.txt.gz', '.pdf', varbvs_file), type='pdf', height=4 * panelheight,
          width=5 * panelheight, units='mm')
    print(p)
    dev.off()
}
ascertainment('/broad/compbio/aksarkar/projects/ctra/results/varbvs-logistic-ascertained.txt.gz', '/broad/compbio/aksarkar/projects/ctra/results/dsvi-logistic-ascertained-1e-3-4000.txt.gz')
ascertainment('/broad/compbio/aksarkar/projects/ctra/results/dsvi-logistic-ascertained-ignore-prevalence.txt.gz', '/broad/compbio/aksarkar/projects/ctra/results/dsvi-logistic-ascertained-1e-3-4000.txt.gz')

bvs_posterior <- function(x, y, t0, t1, sigma, rho, scale) {
    llik <- (sum(dnorm(y, x[,1:2] %*% c(t0, t1), sigma, log=TRUE)) +
             2 * log(0.01) +
             dnorm(t0, 0, scale, log=TRUE) +
             dnorm(t1, 0, scale, log=TRUE))
    if (t0 == 0) {
        llik <- llik + log(0.01) + log(0.99) + dnorm(t1, 0, scale, log=TRUE)
    }
    if (t1 == 0) {
        llik <- llik + log(0.01) + log(0.99) + dnorm(t0, 0, scale, log=TRUE)
    }
    if (t0 == 0 & t1 == 0) {
        llik <- llik + 2 * log(0.99)
    }
    llik
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
    theta0 <- seq(-1.5, 1.5, length.out=100)
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
          labs(title=substitute(paste(h^2, '=', pve, ',', n, '=', n_),
                                list(pve=sprintf('%.3f', pve), n_=dim(Y)[1])),
               x=expression(theta[1]), y=expression(theta[2])) +
          scale_x_continuous(expand=c(0, 0)) +
          scale_y_continuous(expand=c(0, 0)) +
          theme_nature +
          theme(plot.title=element_text(),
                plot.margin=grid::unit(c(0, 2, 0, 0), 'mm')))
    Cairo(file=paste(dir_, 'llik.pdf', sep='/'), type='pdf', width=panelheight,
          height=panelheight, units='mm')
    print(p)
    dev.off()
}

plot_posterior_contour <- function(root) {
    pve <- c('0.005', '0.01', '0.025', '0.05')
    n <- c('5000', '10000', '50000', '100000')
    pi_ <- 0.01
    plots <- (expand.grid(pve, n, pi_) %>%
              dplyr::mutate(dir_=paste(root, sprintf('%s-%s', Var1, Var2), sep='/'), pi_=Var3) %>%
              dplyr::filter(file.exists(dir_)) %>%
              dplyr::group_by(dir_, pi_) %>%
              dplyr::do(plot=posterior_contour(.)))
}
plot_posterior_contour('/broad/hptmp/aksarkar/ctra-evaluate')
