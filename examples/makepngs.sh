#!/bin/bash
for file in ./*
do
    dot -Tpng "$file" > imgs/"$file".png
    echo "$file"
done
