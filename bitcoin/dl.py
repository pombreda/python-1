import sys, urllib, re, urlparse, os.path
from BeautifulSoup import BeautifulSoup

if not len(sys.argv) == 3:
    print >> sys.stderr, 'Usage: %s <URL> <loc>' % (sys.argv[0],)
    sys.exit(1)

url = sys.argv[1]
loc = sys.argv[2]

f = urllib.urlopen(url)
soup = BeautifulSoup(f)

for i in soup.findAll('a', attrs={'href': re.compile('(?i)(csv.gz)$')}):
    full_url = urlparse.urljoin(url, i['href'])
    fname = loc+i['href']
    if not os.path.isfile(fname):
        ret = urllib.urlretrieve(full_url, fname)
        print full_url
