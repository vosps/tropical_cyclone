import os,subprocess


print('running')
model = 'BCC-CSM2-MR'
dir = f'/bp1/geog-tropical/data/CMIP6/CMIP6-rain/{model}/pr/ssp585/'
all_files = os.listdir(dir)
print(all_files)
files = []
for f in all_files:
    if '3hr' in f:
        files.append(f)
file_root = files[0][:-28]
print(file_root)
for file in files:
    print(file)
    year = file[-15:-11]
    print(year)
    subprocess.run(['cdo','splityear',dir+file,dir+file_root+'-split'])

new_files = os.listdir(dir)
new_file_root = new_files[0][:-13]
print(new_file_root)
for new_file in new_files:
    if 'split' in new_file:
        new_file_root = new_file[:-13]
        year = new_file[-7:-3]
        print(new_file)
        print(year)
        print(new_file_root)
        if model == 'MPI-ESM1-2-LR':
            time1 = '01010130'
            time2 = '12312230'
        else:
            time1 = '01010300'
            time2 = '12312100'
        subprocess.run(['mv',dir+new_file,dir+new_file_root+year+time1+'-'+year+time2+ '.nc'])
        print(dir+new_file_root+year+time1+'-'+year+time2+ '.nc')