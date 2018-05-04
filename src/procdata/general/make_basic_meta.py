"""
Scrapes the NHANES website to create an aggregate meta-data table of
all data and meta-data files across all years. Results in GDrive.
"""
import sys
import os
import os.path
import csv
import argparse
from bs4 import BeautifulSoup
import requests

root = os.environ['NHANES_PROJECT_ROOT']
data_dir = os.path.join(root, 'data')

nhanes_master = 'https://wwwn.cdc.gov'
nhanes_source = 'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component={}&CycleBeginYear={}'
years = [year for year in range(1999, 2016, 2)]
comps = ('Demographics', 'Dietary', 'Examination', 'Laboratory', 'Questionnaire', 'Non-Public')
meta_headers = ['Year', 'Comp', 'Data File Name', 'Doc File Name', 'Doc File URL',
'Data File', 'Data File URL', 'Date Published', 'Notes']



def read_meta(args):
    in_path = data_dir
    if not os.path.exists(in_path):
        raise Exception('Csv file not found')
    csv_path = os.path.join(in_path, 'meta_data.csv')
    meta_data = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='~', quotechar='|')
        meta_data = [row for row in reader]
    return meta_data
def gather_meta(args):

    meta_data = [meta_headers]

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
                        cell_url = nhanes_master+cell_content[1]['href']
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
    return meta_data
def get_and_process(meta_data, args, search_set = (years, comps)):
    out_path = data_dir
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    year = ''
    comp = ''
    for row in meta_data[1:]:
        year = row[0]
        comp = row[1]
        if int(year) not in search_set[0] or comp not in search_set[1]: continue
        year_path = os.path.join(out_path, year)
        comp_path = os.path.join(year_path, comp)
        if not os.path.exists(comp_path):
            os.makedirs(comp_path)
        file_url = row[-3]
        try:
            file_path = os.path.join(comp_path, row[3].split()[0]+'.XPT')
            with open(file_path, "wb") as file:
                # get request
                response = requests.get(file_url)
                # write to file
                file.write(response.content)
        except Exception as e:
            print('Issue downloading {} data for {}: {}.'.format(comp, year, str(e)))

def save_meta(meta_data, args):
    out_path = data_dir
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    csv_path = os.path.join(out_path, 'meta_data.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='~',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in meta_data:
            writer.writerow(row)

def main(argv):
    parser = argparse.ArgumentParser(description='NHANES Data Gatherer')
    parser.add_argument('--save-directory', type=str, default='data', help='output directory')
    args = parser.parse_args(argv)

    # meta_data = gather_meta(args)
    # save_meta(meta_data, args)

    meta_data = read_meta(args)
    get_and_process(meta_data, args, (years, ['Dietary']))

if __name__ == '__main__':
    main(sys.argv[1:])
