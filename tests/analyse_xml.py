#from bs4 import BeautifulSoup

#soup = BeautifulSoup(open("fortran/samfshalconv.xml", "r"),"lxml")
#soup.html.body.ofp.file.subroutine.header

from lxml import etree
selector = etree.parse("fortran/samfshalconv.xml")
res = selector.xpath('/ofp/file/subroutine/body/loop')