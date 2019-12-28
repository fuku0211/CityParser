@echo off

cd ..

set sites=kagurazaka nezu_2 tsukishima_3 tsurumaki_5 ushikubo_1
REM set radiuses=0.4 0.7 0.9
REM set min_pts=50 150 200
set configs=0

REM for %%s in (%sites%) do (
REM     for %%r in (%radiuses%) do (
REM         for %%m in (%min_pts%) do (
REM             for %%c in (%configs%) do (
REM                 python src/pointcloud.py cluster -s %%s --radius %%r --min_pts %%m --setting %%c
REM             )
REM         )
REM     )
REM )

for %%s in (%site%) do (
    for %%c in (%config%) do (
        python src/pointcloud.py cluster -s %%s  --setting %%c
    )
)

cd batch