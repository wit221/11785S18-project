import os

if not os.path.exists('data'):
    os.makedirs('data')

if not os.path.exists('data/xpt'):
    os.makedirs('data/xpt')

if not os.path.exists('data/doc'):
    os.makedirs('data/doc')

if not os.path.exists('data/npy'):
    os.makedirs('data/npy')

if not os.path.exists('data/json'):
    os.makedirs('data/json')


cycles = [  '2017-2018',
            '2015-2016',
            '2013-2014',
            '2011-2012',
            '2009-2010',
            '2007-2008',
            '2005-2006',
            '2003-2004',
            '2001-2002',
            '1999-2000']

for c in cycles:

    if not os.path.exists('data/xpt/'+c):
        os.makedirs('data/xpt/'+c)

    if not os.path.exists('data/doc/'+c):
        os.makedirs('data/doc/'+c)
