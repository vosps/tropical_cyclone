import os,subprocess
print('running')
dir = '/user/home/al18709/work/CMIP6/TaiESM/pr/ssp585/'
files = os.listdir(dir)
print(files)
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

        subprocess.run(['mv',dir+new_file,dir+new_file_root+year+'01010300-'+year+'12312100.nc'])