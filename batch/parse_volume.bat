@echo off

cd ..

set sites=kagurazaka nezu_2 tsukishima_3 tsurumaki_5 ushikubo_1
set configs=0

for %%s in (%sites%) do (
    for %%c in (%configs%) do (
        python src/parse.py volume -s %%s --setting %%c
    )
)

cd batch