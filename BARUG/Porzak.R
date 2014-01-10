# Data profiling with R
# Jim Porzak
# Viadeo

# BARUG July 2011
grid.newpage()

pname <- paste("Plots/ColPlot_", ColDesc$Table, '_C', ithCol, '.png', sep = '')
png(pname, width = 10.5, height = 3, units = 'in', res = 600, pointsize = 10)

TopLayout <- grid.layout(nrow = 3, ncol = 3,
                         widths = unit(c(3, 1,2), c('null', 'null')),
                         heights = unit(c(2, 1, 3), c('lines', 'null', 'lines'))
                         )

# grid.show.layout(TopLayout)  ## For debugging
vpTopLayout <- viewport(layout = TopLayout)

library(ggplot2)


# Grid Graphics Tricks (2)
pushViewport(vpTopLayout)
grid.rect(gp = gpar(col = 'blue', lwd = 3, fill = 'NA'))

pushViewport(viewport(layout.pos.col = 1:2, layout.pos.row = 1))
grid.rect(gp = gpar(col = 'green', lwd = 2))

grid.text(paste('t1', 't2', 't3', sep = ' . '),
          x = unit(0.2, 'char'), y = unit(0.6, 'lines'), just = 'left',
          gp = gpar(col = 'black', fontsize = 18))

grid.text('csn', x = unit(0.8, 'npc'), y = unit(0.6, 'lines'),
          just = 'right', gp = gpar(col = 'black', fontsize = 18))

grid.text(paste(ColDesc$ColType, '(',
                ifelse(is.na(colDesc$ColWidth), '', ColDesc$ColWidth),
                ') ', sep = ''),
          x = unit(1, 'npc'), y = unit(0.6, 'lines'), just = 'right', 
          gp = gpar(col = 'black', fontsize = 18))

popViewport()



# Grid graphics tricks (3)
### generate plot as function of ColPlot value
p <- NULL  ## one of the following should build a plot p, if not throw error plot
# a Category plot
if (ColDesc$ColPlot == 'Category') {
  p <- qplot(ColValue, NumRows, data = PlotValues, geom = 'bar',
             stat = 'identity', xlab = '', ylab = '# Rows',
             main = paste('Categories in', ColDesc$Column),
             fill = I('grey50')) + coord_flip()
}

### Cut ###

# throw error plot if p not built above
if (is.null(p)) p <- 'No plot was built'

## lastly output ggplot generated above as p
pushViewport(viewport(layout.pos.col = 2, layout.pos.row = 2:3))
print(p, vp = viewport(layout.pos.row = 2, layout.pos.col = 2))
dev.off()


### Bartev tried this
library(ggplot2)
grid.newpage()

midLayout <- grid.layout(nrow = 2, ncol = 2,
                         widths=unit(c(3, 2), 'null'),
                         )
TopLayout <- grid.layout(nrow = 3, ncol = 3,
                         widths = unit(c(3, 1,2), c('null', 'null')),
                         heights = unit(c(2, 1, 3), c('lines', 'null', 'lines'))
                         )

quadLayout <- grid.layout(nrow = 2, ncol = 2,
                          widths = unit(c(3, 2), 'null'),
                          heights = unit(c(2, 1), c('lines', 'null')))
# current.viewport()
# current.vpTree()
grid.show.layout(quadLayout)  ## For debugging
vpMidLayout <- viewport(layout = midLayout)
# pushViewport(vpMidLayout)
current.column()
current.viewport()



grid.newpage()
pushViewport(vpMidLayout)
pushViewport(viewport(layout.pos.col = 1))
grid.rect(gp = gpar(col = 'blue', lwd = 2))

upViewport(1)
pushViewport(viewport(layout.pos.col = 2))
grid.rect(gp = gpar(col = 'red', lwd = 2))

print(p, vp=viewport(layout.pos.row = 1, layout.pos.col = 2))

p <- qplot(mpg, disp, data = mtcars, geom='point') + facet_wrap(~ cyl)
p2 <- qplot(mpg, disp, data = mtcars, geom='point') + facet_grid(cyl ~ gear)
upViewport(1)
pushViewport(viewport(layout.pos.col = 1))

print(p2, vp=viewport(layout.pos.row = 1, layout.pos.col = 1))
grid.rect(gp=gpar(fill = 'grey20'))
print(p2)

grid.newpage()

pushViewport(vpMidLayout)
pushViewport(viewport(layout.pos.col = 2))
grid.rect(gp = gpar(col = 'blue', lwd = 2))
current.viewport()
current.vpPath()
current.viewport()
current.vpTree()
upViewport(1)
current.viewport()
current.vpPath()
pushViewport(viewport(layout.pos.col = 1))
current.vpPath()
print(p2, vp=viewport(layout.pos.row=1, layout.pos.col = 2))




pushViewport(viewport(layout.pos.col = 1))  ## activate col 1
grid.rect(gp = gpar(col = 'blue', lwd = 2,vp=viewport(layout.pos.col = 1)))  ## draw blue rect
upViewport(1)  ## go up to vp with 2 cols
pushViewport(viewport(layout.pos.col = 1))  ## activate col 1
upViewport(1)  ## go up to vp with 2 cols
pushViewport(viewport(layout.pos.col = 2))  ## activate col 2
grid.rect(gp = gpar(col = 'red', lwd = 2), vp=viewport(layout.pos.col = 2))  ## draw red rect
current.viewport()



## Draw 2 different plots to different windows
p <- qplot(mpg, disp, data = mtcars, geom='point') + facet_grid(cyl ~ .)
p2 <- qplot(mpg, disp, data = mtcars, geom='point') + facet_wrap(~ cyl)

grid.newpage()  ## clear page
pushViewport(vpMidLayout)  ## give vp that I previously defined
print(p, vp=viewport(layout.pos.row=2, layout.pos.col = 1))  ## print graphic to col 1
print(p2, vp=viewport(layout.pos.row=2, layout.pos.col = 2))  ## print graphic to col 2