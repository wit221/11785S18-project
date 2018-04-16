"""
Given NHANES master data website, for each year:
    -create year folder
    -create data cat folders for each cat
        -download all files in cat
        -for each file
            -produce meta data
            -convert to csv/numpy
        -possibly: merge all files for a cat in one master file
    -possibly: merge all files for a year in one master file
-possibly: merge all files for all years in one master file

"""
import argparse
from bs4 import BeautifulSoup
import requests

def gather(args):
    nhanes_source = 'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component={}&CycleBeginYear={}'
    years = [year for year in range(1999, 2018, 2)]
    comps = ('Demographics', 'Dietary', 'Examination', 'Laboratory', 'Questionnaire', 'Non-Public')

    meta_headers = [['Year', 'Comp', 'Data File Name', 'Doc File Name', 'Doc File URL',
     'Data File', 'Data File URL', 'Date Published', 'Notes']]

    meta_data = meta_headers

    for year in years:
        for comp in comps:
            url = nhanes_source.format(comp, year)
            r  = requests.get(url)
            data = r.text
            soup = BeautifulSoup(data, 'lxml')
            table = soup.find('table', {'id':'GridView1'})

            try:
                rows = table.find_all('tr')[1:]
            except Exception as e:
    #             print('Issue fetching {} data for {}. Data unavailable.'.format(comp, year))
                meta_data.append([str(year), comp] + ['']*(len(meta_data)-3)+['Unavailable'])
                continue
            for row in rows:
                note = ''
                cell_contents = [cell.contents for cell in row.find_all('td')]
                cells = [str(year), comp]
                for cell_content in cell_contents:
                    if len(cell_content)==1:
                        try:
                            to_append = cell_content[0].strip()
                            cells.append(to_append)
                            if to_append == 'RDC Only': cells.append('')
                        except Exception as e:
                            print('Issue parsing {} data for {}. Data likely withdrawn.'.format(comp, year))
                            to_append = cell_content[0].contents[0].strip()
                            if len(to_append) == 0:
                                try:
                                    alert = cell_content[0].find('a')['onclick']
                                    note = alert.split("('")[1].split("')")[0]
                                except Exception as e:
                                    pass
                            cells.append(to_append.strip())
                    elif len(cell_content) == 3:
                        cell_url = cell_content[1]['href']
                        cell_url_name = cell_content[1].contents[0]
                        cells.append(cell_url_name)
                        cells.append(cell_url)
                    else:
                        raise Exception('Unrecognized cell type')

                if len(note)==0:
                    cells += ['']
                else:
                    cells += ['']*2+[note]
                meta_data.append(cells)
def main(argv)
    parser = argparse.ArgumentParser(description='PyTorch GAN Example')
    parser.add_argument('--save-directory', type=str, default='output', help='output directory')
    args = parser.parse_args(argv)
    gather(args)

if __name__ == '__main__':
    main(argv[1:])
