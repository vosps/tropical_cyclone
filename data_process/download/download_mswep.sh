#!/bin/bash
dataDir=/bp1store/geog-tropical/data/Obs/MSWEP/3hourly
cd $dataDir
# for year in {2020..2022}; do # need to do 1970 from day 032
# for day in 00{1..9} 0{10..99} {100..366}; do # this should be all done simultaneously
# for hour in 00 03 06 09 12 15 18 21; do
for year in {2021..2022}; do # need to do 1970 from day 032
for day in 00{1..9} 0{10..99} {100..366}; do # this should be all done simultaneously
for hour in 00 03 06 09 12 15 18 21; do
echo "google_drive:/MSWEP_V280/Past/3hourly/"$year$day"."$hour".nc"
rclone sync --drive-shared-with-me "google_drive:/MSWEP_V280/Past/3hourly/"$year$day"."$hour".nc" ./.  ; done ; done & done
echo "saved"
# rclone sync --drive-shared-with-me "GoogleDrive:/MSWEP_V280/Past/3hourly/"$year$day"."$hour".nc" ./ .  ; done ; done & done