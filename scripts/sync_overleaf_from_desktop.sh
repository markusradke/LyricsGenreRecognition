#!/bin/bash

DOWNLOADED_PATH="/mnt/c/Users/marku/Desktop/Power_to_the_People__Assigning_Popular_Music_Genres_Based_on_Listeners__Perspective_Using_a_Genre_Tree_Approach.zip"

REPORT_PATH="/mnt/n/Materialien/Promotion/GenreDiscourseAnalysis/reports/paper_v1/"

if [ -f $DOWNLOADED_PATH ]; then
  echo "Unzipping file to project folder..."
  unzip -o $DOWNLOADED_PATH -d $REPORT_PATH
  rm $DOWNLOADED_PATH
  echo "Done."
else
  echo "Downloaded File does not exist. Please export the file from Overleaf with 'File > Download as source (.zip)' Aborting script..."
fi
