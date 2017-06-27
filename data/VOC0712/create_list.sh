#!/bin/bash
caffe_root=/cygdrive/e/code/caffe/caffe_ssd_windows
#root_dir=$HOME/data/VOC/VOCdevkit/
root_dir=$caffe_root/data/VOC0712/VOCdevkit/
sub_dir=ImageSets/Main
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for dataset in trainval test
do
  dst_file=$bash_dir/$dataset.txt
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  for name in VOC2007 VOC2012
  do
    if [[ $dataset == "test" && $name == "VOC2012" ]]
    then
      continue
    fi
    echo "Create list for $name $dataset..."
    dataset_file=$root_dir/$name/$sub_dir/$dataset.txt

    img_file=$bash_dir/$dataset"_img.txt"
    cp $dataset_file $img_file
    sed -i "s/^/$name\/JPEGImages\//g" $img_file
    sed -i "s/$/.jpg/g" $img_file

    label_file=$bash_dir/$dataset"_label.txt"
    cp $dataset_file $label_file
    sed -i "s/^/$name\/Annotations\//g" $label_file
    sed -i "s/$/.xml/g" $label_file

    paste -d' ' $img_file $label_file >> $dst_file

    rm -f $label_file
    rm -f $img_file
  done
  echo "End create list for $dataset."
  echo "bash dir: $bash_dir, root dir: $root_dir, dst file: $dst_file"
  # Generate image name and size infomation.
  if [ $dataset == "test" ]
  then
    # to run windows built exe "get_image_size.exe", all path string in parameters should be windows style path
    #$bash_dir/../../ssd/tools/release/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
    root_dir_win=$(cygpath -C ANSI -w "$root_dir")
    dst_file_win=$(cygpath -C ANSI -w "$dst_file")
    bash_dir_win=$(cygpath -C ANSI -w "$bash_dir")
    #echo "$root_dir_win, $dst_file_win, $bash_dir_win"
    $bash_dir/../../ssd/tools/release/get_image_size $root_dir_win $dst_file_win $bash_dir_win/$dataset"_name_size.txt"
  fi

  # Shuffle trainval file.
  if [ $dataset == "trainval" ]
  then
    rand_file=$dst_file.random
    echo "Rand file: $rand_file"
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
  fi
done
