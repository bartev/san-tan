# Based on Jim Porzak's talk at BARUG july 2011
# drawProfile <- function(df, cn)
# Will create 4 regions
# Top: column name, class
# Middle left: summary stats
# Middle right: plot
# Bottom left: head(column)


# Use viewports to draw several rectangles
ProfileDataFrame <- function (df, plotname = 'Profile_Plot') {
  #   pname <- 'Plots_Summary %d.png'
  # Run drawProfile(df, c) over all columns of a dataframe
  # Produces png files numbered 1-n
  for (c in seq_along(df)) {
    cname <- dimnames(df)[[2]][c]
    pname <- paste(plotname, c, cname, sep = ' ')
    pname <- paste(pname, '.png', sep = '')
    png(pname, width = 7, height = 2, units = 'in', res = 600, pointsize = 8)
    drawProfile(df, c)
    dev.off()  
  }
  #   Could put for loop inside, but then can't get column names on the files
  #   pname <- paste(plotname, '_%d.png', sep = ' ')
  #   png(...
  #   for ...
  #   dev.off()
}

###### start here ####
# drawProfile(df, 9)
# drawProfile(myList, mtcars, 1)

## Helper functions
## Create the plot
gp <- function(df, cn) {
  require(ggplot2)
  p <- ggplot(df, aes_string(x = names(df)[cn]), aes(fill='red')) + geom_bar(fill = 'red')
}

## Create list (5 col) summary
csl <- function(df, cn) {
  require(stringr)
  require(grid)
#   saveDigits <- options('digits')
#   options('digits' = 4)
  # Column Summary List
  cname <- dimnames(df)[[2]][cn]
  
  NumRows <- nrow(df[cname])
  Nulls <- sum(is.na(df[cname]))
  Distinct <- nrow(unique(df[cname]))
  
  ss <- summary(df[cname], na.rm = TRUE)  ## get summary (results in table)
  ss1 <- sapply(ss, str_split, ':')  ## split label from value (results in list)
  ss1.t <- sapply(ss1, str_trim)  ## trim spaces (results in matrix)
  
  list1 <- paste('Rows', NumRows, round(100.00 * NumRows / NumRows, digits=4), sep = '\n')
  list2 <- paste('Nulls', Nulls, round(100.00 * Nulls / NumRows, digits=4), sep = '\n', collapse = '')
  list3 <- paste('Distinct', Distinct, round(100 * Distinct / NumRows, digits=4), sep = '\n')
  list4 <- paste(ss1.t[1, ], collapse = '\n')
  list5 <- paste(ss1.t[2, ], collapse = '\n')

  if (class(df[,cn]) == 'factor') {
    list4 <- paste('Factor\n', list4, sep = '')
    list5 <- paste('Freq\n', list5, sep = '')
  }
#   options(saveDigits)
  myList <- list(list1, list2, list3, list4, list5)
}

## Main function - for one column
drawProfile <- function(df, cn) {
  # create plot from df and column number
  p <- gp(df, cn)
  
  # Create summary list
  strList <- csl(df, cn)
  
  #   Will draw text and plot in 2 different regions of the drawing device
  grid.newpage()
  quadLayout <- grid.layout(nrow = 3, ncol = 2,
                            widths = unit(c(3, 2), 'null'),
                            heights = unit(c(2, 1, 2), c('lines', 'null', 'lines')))
  # grid.show.layout(quadLayout)  ## For debugging
  vpQuadLayout <- viewport(layout = quadLayout)
  
  # current.viewport()  ## for debugging
  pushViewport(vpQuadLayout)
  
  # draw blue box around top row (both columns)
  pushViewport(viewport(layout.pos.col=1:2, layout.pos.row=1))
  grid.rect(gp=gpar(col = 'blue', lwd = 2))
  popViewport()
  
  # draw red box around r2, c1
  pushViewport(viewport(layout.pos.col=1, layout.pos.row=2))
  grid.rect(gp=gpar(col = 'red', lwd = 2))
  popViewport()
  
  # draw green box around around r2:r3, c2
  pushViewport(viewport(layout.pos.col=2, layout.pos.row=2:3))
  grid.rect(gp=gpar(col = 'green', lwd = 2))
  popViewport()
  
  # draw black box around around r3, c1
  pushViewport(viewport(layout.pos.col=1, layout.pos.row=3))
  grid.rect(gp=gpar(col = 'black', lwd = 2))
  popViewport()
  
  
  # after each popViewport(), back to vpQuadLayout level
  print(p, vp=viewport(layout.pos.row=2, layout.pos.col = 2))  ## print graphic to col 2
  
  # draw some text in left box
  pushViewport(viewport(layout.pos.col=1, layout.pos.row=2))
  # col 1, #, %
  t1 <- '\n#\n%'
  grid.text(t1,
            x = unit(1, 'lines'), y = unit(1, 'npc') - unit(1, 'lines'), just = c('left', 'top'), 
            gp = gpar(col = 'black', fontsize = 8))
  # Col 2 Rows
  for (i in seq_along(strList)) {
    grid.text(strList[[i]],
              x = unit(-1, 'picas') + unit(4 * i, 'picas'), y = unit(1, 'npc') - unit(1, 'lines'), just = c('left', 'top'), 
              gp = gpar(col = 'black', fontsize = 8))
  }
  popViewport()
  
  # Header text
  # Create header
  txtHeaderLeft <- paste('ColName: ', dimnames(df)[[2]][cn], sep = '')
  txtHeaderRight <- paste('Class: ', class(df[ ,cn]), sep = '')
  
  pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 1:2)) ## print text to top row
  grid.text(txtHeaderLeft,
            x = unit(1, 'lines'), y = unit(0.5, 'npc'), just = 'left',
            gp = gpar(col = 'blue', fontsize = 12))
  grid.text(txtHeaderRight,
            x = unit(1, 'npc') - unit(1, 'lines'), y = unit(0.5, 'npc'), just = 'right',
            gp = gpar(col = 'blue', fontsize = 12))
  popViewport()
  
  # Footer text
  # Create footer
  txtFooter <- toString(head(df[cn]))
  txtFooter <- paste('Head: ', txtFooter, sep = '')
  pushViewport(viewport(layout.pos.row = 3, layout.pos.col = 1))
  grid.text(txtFooter,
            x = unit(1, 'lines'),  y = unit(0.9, 'npc'), just = c('left', 'top'),
            gp = gpar(col = 'black', fontsize = 6))
  popViewport()
}


