import os
import urllib
import urllib2
import threading

#from xml.dom import minidom, Node
import xml.etree.ElementTree as ET

def getcat(cat):
    catfn = '%s.cat.xml' % cat
    if not os.path.exists(catfn):
        fn = '%s.dd.xml' % cat
        if not os.path.exists(fn):
            url = 'http://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd?mode=xml&short=0&catalog=%s' % cat
            print 'Retrieving', url
            f = urllib2.urlopen(url)
            data = f.read()
            f = open(fn, 'w')
            f.write(data)
            f.close()
            print 'Wrote', len(data), 'to', fn
    
        tree = ET.parse(fn)
        root = tree.getroot()
        print 'Parsed XML:', root
    
        colnames = []
        for col in root.iter('column'):
            name = col.find('colname')
            print '  column:', name.text
            colnames.append(name.text)
    
        url = 'http://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?outfmt=3&spatial=NONE&catalog=%s&selcols=%s' % (cat, ','.join(colnames))
        print 'Retrieving URL', url
        print 'URL length', len(url)
    
        # f = urllib2.urlopen(url, None, 1000000)
        # data = f.read()
        # f = open(catfn, 'w')
        # f.write(data)
        # f.close()
        # print 'Wrote', len(data), 'to', catfn

        # cmd = 'wget --timeout 0 -O %s "%s"' % (catfn, url)
        # print 'Running:', cmd
        # os.system(cmd)

        RR = range(0, 360+1, 10)
        fns = []
        for i,(rlo,rhi) in enumerate(zip(RR, RR[1:])):
            constraint = 'ra>%g&ra<=%g' % (rlo,rhi)
            url2 = url + '&' + urllib.urlencode(dict(constraint=constraint))
            print 'url2', url2
            catfn2 = catfn + '-%i' % i
            cmd = 'wget --timeout 0 -O %s "%s"' % (catfn2, url2)
            print 'Running:', cmd
            os.system(cmd)
            fns.append(catfn2)

        cmd = 'cat %s > %s' % (' '.join(fns), catfn)
        print 'Running:', cmd
        os.system(cmd)

    tree = ET.parse(catfn)
    root = tree.getroot()
    print 'Parsed XML:', tree

    tab = root.find('table')
    fieldnames = []
    fieldtypes = []
    typemap = dict(char=str, int=np.int64, double=np.float64, float=np.float64)
    for f in tab.findall('field'):
        print 'Field', f.attrib
        nm = f.attrib['name']
        ty = f.attrib['datatype']
        fieldnames.append(nm)
        fieldtypes.append(typemap[ty])

    data = [[] for f in fieldnames]
    
    datanode = root.find('tabledata')
    for tr in datanode:
        assert(len(tr) == len(fieldtypes))
        for td,typ,dd in zip(tr, fieldtypes, data):
            dd.append(typ(td.text))

    data = [np.array(dd, dtype=tt) for dd,tt in zip(data, fieldtypes)]

    T = tabledata()
    for dd,nn in zip(data, fieldnames):
        T.set(nn, dd)
    T.about()
    ffn = '%s.fits' % cat
    T.writeto(ffn)
    print 'Wrote', ffn


threads = []
for cat in ['wise_allsky_2band_p1bs_frm',
            'wise_allsky_3band_p1bs_frm',
            'wise_allsky_4band_p1bs_frm',
            ]:
    t = threading.Thread(target=getcat, args=(cat,))
    t.start()
    threads.append(t)
for t in threads:
    print 'Waiting for threads to finish...'
    t.join()
print 'Done!'
