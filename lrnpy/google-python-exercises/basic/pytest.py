#!usr/bin/python

import urllib

## Given a url, try to retrieve it. If it's text/html,
## print its base url and its text.
def wget(url):
  ufile = urllib.urlopen(url)  ## get file-like object for url
  info = ufile.info()   ## meta-info about the url content
  if info.gettype() == 'text/html':
    print 'base url:' + ufile.geturl()
    text = ufile.read()  ## read all its text
    print text[:200]

wget('https://www.facebook.com/')


## Given a url, try to retrieve it. If it's text/html,
## print its base url and its text.
## Version that uses try/except to print an error message if the
## urlopen() fails.
def wget(url):
  try:
    ufile = urllib.urlopen(url)
    if ufile.info().gettype() == 'text/html':
      print ufile.read()
  except IOError:
    print 'problem reading url:', url
   
str = """10.254.254.138 - - [06/Aug/2007:00:05:31 -0700] "GET /edu/languages/google-python-class/images/puzzle/a-baaj.jpg HTTP/1.0" 302 4067 "-" "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.6) Gecko/20070725 Firefox/2.0.0.6"
"""
    