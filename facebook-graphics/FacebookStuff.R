# 2012-01-25
# Crawling FB with R
# See websites:
# http://applyr.blogspot.com/2012/01/mining-facebook-data-most-liked-status.html?spref=tw
# http://romainfrancois.blog.free.fr/index.php?post/2012/01/15/Crawling-facebook-with-R


setwd("~/Rmac/FacebookGraphics")


# Graph API Explorer
# https://developers.facebook.com/tools/explorer?method=GET&path=673561182


access_token <- "AAACEdEose0cBAGMV4n3xi1wzQaax5KYEQW0HzjE7O4QRffH0ZCEB0xoRcasMIbTR0tzUjoGiZB9rrZBvAcmDqpddJ7tMJag7pnBDP2QEwZDZD"

require(RCurl)
require(rjson)

facebook <- function(path='me', access_token=token, options) {
  if (!missing(options) ){
    options <- sprintf( "?%s", paste( names(options), "=", unlist(options), collapse = "&", sep = ""))
  } else {
    options <- ""
  }
  data <- getURL( sprintf( "https://graph.facebook.com/%s%s&access_token=%s", path, options, access_token))
  fromJSON( data )
}

dir.create("photos")
photos <- facebook("me/photos", access_token)
sapply( photos$data, function(x) {
  url <- x$source
  download.file( url, file.path( "photos", basename(url)))
})


# Scrape the list of friends
friends <- facebook( path='me/friends', access_token=access_token)
# Extract Facebook IDs
friends.id <- sapply(friends$data, function(x) x$id)

# Do a similar thing using plyr
# require(plyr)
# friends.id.2 <- ldply(friends$data, function(x) { cbind(x$name, x$id)})
# names(friends.id.2) <- c('Name', 'id')

# Extract names
friends.name <- sapply(friends$data, function(x) iconv(x$name, 'UTF-8', 'ASCII//TRANSLIT'))
# Short names to initials
initials <- function(x) paste(substr(x, 1, 1), collapse="")
friends.initial <- sapply(strsplit(friends.name, " "), initials)
head(friends.initial)

# friendship relation matrix
N <- length(friends.id)
friendship.matrix <- matrix(0,N,N)
for (i in 1:N) {
  tmp <- facebook( path=paste("me/mutualfriends", friends.id[i], sep="/") , access_token=access_token)
  mutualfriends <- sapply(tmp$data, function(x) x$id)
  friendship.matrix[i,friends.id %in% mutualfriends] <- 1
}

# install.packages('Rgraphviz')
require(Rgraphviz)

# convert relation matrix to graph
g <- new('graphAM', adjMat=friendship.matrix)


# ellipse graph with initials
pdf(file='facebook1.pdf', width=25, height=25)
  attrs <- list(node=list(shape='ellipse', fixedsize=FALSE))
  nAttrs <- list(label=friends.initial)
  names(nAttrs$label) <- nodes(g)
  plot(g, 'neato', attrs=attrs, nodeAttrs=nAttrs)
dev.off()

# install.packages('pixmap')
require(pixmap)
# download small profile picture of each friend
for (i in 1:length(friends.id))
  download.file(paste("http://graph.facebook.com", friends.id[i], "picture", sep="/"),
                destfile=paste("photos/", friends.id[i], ".jpg", sep=""))
# mac osx doesn't have shell command 'convert'
# system('for i in `ls photos/*.jpg`; do j=${i%.*}; convert $j.jpg $j.pnm; done', wait=TRUE)

# customized node plotting function
install.packages("rimage")
install.packages("ReadImages")
library(ReadImages)
# no way to convert from jpg to pnm, so this function isn't working quite yet.
makeNodeDrawFunction <- function(x) {
  force(x)
  function(node, ur, attrs, radConv) {
    photo <- read.jpeg(paste("photos/", x, ".jpg", sep=""))
    nc <- getNodeCenter(node)
    addlogo(photo, c(getX(nc)-25, getX(nc)+25), c(getY(nc)-25, getY(nc)+25))
  }
}
drawFuns <- apply(as.array(friends.id), 1, makeNodeDrawFunction)

# a graph with photos
pdf(file="facebook2.pdf", width=25, height=25)
  attrs <- list(node=list(shape='box', width=0.75, height=0.75))
  plot(g, 'neato', attrs=attrs, drawNode=drawFuns)
dev.off()