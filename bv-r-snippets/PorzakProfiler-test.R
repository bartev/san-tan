
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

# ProfileDataFrame(df)