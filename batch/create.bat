@echo off

cd ..

set sites=kagurazaka nezu_2 tsukishima_3 ushikubo_1 tsurumaki_5

for %%s in (%sites%) do (
    python src/pointcloud.py create -s %%s
    python src/pointcloud.py create -s %%s --with_seg
)

cd batch