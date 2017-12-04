## arg1: input dir
## arg2: output dir
arg1=/home/ubuntu/jd_ai/row_data/train
arg2=/home/ubuntu/jd_ai/data3/train

suffix=.mp4
for filename in $(ls $arg1)
do
	outname="${filename%$suffix}"
	mkdir $arg2/$outname
	ffmpeg -i $arg1/$filename -aspect 16:9 -r 20 $arg2/$outname/%04d.bmp
done