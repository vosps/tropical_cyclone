#!/bin/bash
#SBATCH --time=0-6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=36gb
#SBATCH --cpus-per-task=1
#SBATCH --job-name=download_mswep
#SBATCH --account=GEOG022743
#SBATCH --partition dmm



# dataDir=/bp1/geog-tropical/data/Obs/MSWEP/3hourly
# cd $dataDir
# # for year in {2020..2022}; do # need to do 1970 from day 032
# # for day in 00{1..9} 0{10..99} {100..366}; do # this should be all done simultaneously
# # for hour in 00 03 06 09 12 15 18 21; do
# for year in {2021..2022}; do # need to do 1970 from day 032
# for day in 00{1..9} 0{10..99} {100..366}; do # this should be all done simultaneously
# for hour in 00 03 06 09 12 15 18 21; do
# echo "google_drive:/MSWEP_V280/Past/3hourly/"$year$day"."$hour".nc"
# rclone sync --drive-shared-with-me "google_drive:/MSWEP_V280/Past/3hourly/"$year$day"."$hour".nc" ./.  ; done ; done & done
# echo "saved"
# rclone sync --drive-shared-with-me "GoogleDrive:/MSWEP_V280/Past/3hourly/"$year$day"."$hour".nc" ./ .  ; done ; done & done

dataDir=/bp1/geog-tropical/data/Obs/MSWEP/missing
cd $dataDir
# for year in {2020..2022}; do # need to do 1970 from day 032
# for day in 00{1..9} 0{10..99} {100..366}; do # this should be all done simultaneously
# for hour in 00 03 06 09 12 15 18 21; do

# Read the file names from the text file into a Bash array
mapfile -t files < /user/home/al18709/tropical_cyclones/chapter_3/file_list.txt

# Iterate over the file names
for file in "${files[@]}"
do
    # Do whatever you want with each file, for example:
    echo "Processing $file"
    echo "google_drive:/MSWEP_V280/Past/3hourly/"$file""
    rclone sync --drive-shared-with-me "google_drive:/MSWEP_V280/Past/3hourly/"$file"" ./. 
    echo "saved"
    # Add your processing logic here
done

echo "google_drive:/MSWEP_V280/Past/3hourly/"$year$day"."$hour".nc"
rclone sync --drive-shared-with-me "google_drive:/MSWEP_V280/Past/3hourly/"$year$day"."$hour".nc" ./.  ; done ; done & done
echo "saved"